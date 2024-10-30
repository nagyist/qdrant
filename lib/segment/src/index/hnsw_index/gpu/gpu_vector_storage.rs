use std::borrow::Cow;
use std::sync::Arc;

use common::types::PointOffsetType;
use quantization::{EncodedStorage, EncodedVectorsPQ, EncodedVectorsU8};

use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::primitive::PrimitiveVectorElement;
use crate::data_types::vectors::{VectorElementType, VectorElementTypeByte, VectorElementTypeHalf};
use crate::index::hnsw_index::gpu::GPU_TIMEOUT;
use crate::types::Distance;
use crate::vector_storage::quantized::quantized_vectors::{
    QuantizedVectorStorage, QuantizedVectors,
};
use crate::vector_storage::{DenseVectorStorage, VectorStorage, VectorStorageEnum};

pub const ELEMENTS_PER_SUBGROUP: usize = 4;
pub const UPLOAD_CHUNK_SIZE: usize = 64 * 1024 * 1024;
pub const STORAGES_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuVectorStorageElementType {
    Float32,
    Float16,
    Uint8,
    Binary,
    SQ,
    PQ,
}

pub struct GpuVectorStorage {
    pub device: Arc<gpu::Device>,
    pub vectors_buffer: Vec<Arc<gpu::Buffer>>,
    pub descriptor_set_layout: Arc<gpu::DescriptorSetLayout>,
    pub descriptor_set: Arc<gpu::DescriptorSet>,
    pub dim: usize,
    pub count: usize,
    pub element_type: GpuVectorStorageElementType,
    pub distance: Distance,
    pub quantization: Option<GpuQuantization>,
}

pub enum GpuQuantization {
    Scalar(GpuScalarQuantization),
    Product(GpuProductQuantization),
}

pub struct GpuScalarQuantization {
    pub multiplier: f32,
    pub diff: f32,
    pub offsets_buffer: Arc<gpu::Buffer>,
}

pub struct GpuProductQuantization {
    pub centroids_buffer: Arc<gpu::Buffer>,
    pub vector_division_buffer: Arc<gpu::Buffer>,
    pub divisions_count: usize,
}

impl GpuVectorStorage {
    pub fn new(
        device: Arc<gpu::Device>,
        vector_storage: &VectorStorageEnum,
        quantized_storage: Option<&QuantizedVectors>,
        force_half_precision: bool,
    ) -> OperationResult<Self> {
        if let Some(quantized_storage) = quantized_storage {
            Self::new_from_vector_quantization(
                device,
                vector_storage,
                &quantized_storage.storage_impl,
            )
        } else {
            Self::new_from_vector_storage(device, vector_storage, force_half_precision)
        }
    }

    fn new_from_vector_quantization(
        device: Arc<gpu::Device>,
        vector_storage: &VectorStorageEnum,
        quantized_storage: &QuantizedVectorStorage,
    ) -> OperationResult<Self> {
        match quantized_storage {
            QuantizedVectorStorage::ScalarRam(quantized_storage) => {
                Self::new_typed::<VectorElementTypeByte>(
                    device.clone(),
                    GpuVectorStorageElementType::SQ,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| {
                        let (_, vector) = quantized_storage.get_quantized_vector(id);
                        Cow::Borrowed(vector)
                    },
                    Some(GpuQuantization::Scalar(GpuScalarQuantization {
                        multiplier: quantized_storage.get_multiplier(),
                        diff: quantized_storage.get_diff(),
                        offsets_buffer: Self::create_sq_offsets_buffer(
                            device,
                            &quantized_storage,
                            vector_storage.total_vector_count(),
                        )?,
                    })),
                )
            }
            QuantizedVectorStorage::ScalarMmap(quantized_storage) => {
                Self::new_typed::<VectorElementTypeByte>(
                    device.clone(),
                    GpuVectorStorageElementType::SQ,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| {
                        let (_, vector) = quantized_storage.get_quantized_vector(id);
                        Cow::Borrowed(vector)
                    },
                    Some(GpuQuantization::Scalar(GpuScalarQuantization {
                        multiplier: quantized_storage.get_multiplier(),
                        diff: quantized_storage.get_diff(),
                        offsets_buffer: Self::create_sq_offsets_buffer(
                            device,
                            &quantized_storage,
                            vector_storage.total_vector_count(),
                        )?,
                    })),
                )
            }
            QuantizedVectorStorage::PQRam(quantized_storage) => {
                Self::new_typed::<VectorElementTypeByte>(
                    device.clone(),
                    GpuVectorStorageElementType::PQ,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| {
                        let vector = quantized_storage.get_quantized_vector(id);
                        Cow::Borrowed(vector)
                    },
                    Some(GpuQuantization::Product(Self::create_pq_quantization(
                        device,
                        quantized_storage,
                    )?)),
                )
            }
            QuantizedVectorStorage::PQMmap(quantized_storage) => {
                Self::new_typed::<VectorElementTypeByte>(
                    device.clone(),
                    GpuVectorStorageElementType::PQ,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| {
                        let vector = quantized_storage.get_quantized_vector(id);
                        Cow::Borrowed(vector)
                    },
                    Some(GpuQuantization::Product(Self::create_pq_quantization(
                        device,
                        quantized_storage,
                    )?)),
                )
            }
            QuantizedVectorStorage::BinaryRam(quantized_storage) => {
                Self::new_typed::<VectorElementTypeByte>(
                    device,
                    GpuVectorStorageElementType::Binary,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| Cow::Borrowed(quantized_storage.get_quantized_vector(id)),
                    None,
                )
            }
            QuantizedVectorStorage::BinaryMmap(quantized_storage) => {
                Self::new_typed::<VectorElementTypeByte>(
                    device,
                    GpuVectorStorageElementType::Binary,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| Cow::Borrowed(quantized_storage.get_quantized_vector(id)),
                    None,
                )
            }
            QuantizedVectorStorage::ScalarRamMulti(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            QuantizedVectorStorage::ScalarMmapMulti(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            QuantizedVectorStorage::PQRamMulti(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            QuantizedVectorStorage::PQMmapMulti(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            QuantizedVectorStorage::BinaryRamMulti(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            QuantizedVectorStorage::BinaryMmapMulti(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
        }
    }

    fn new_from_vector_storage(
        device: Arc<gpu::Device>,
        vector_storage: &VectorStorageEnum,
        force_half_precision: bool,
    ) -> OperationResult<Self> {
        match vector_storage {
            VectorStorageEnum::DenseSimple(vector_storage) => {
                if force_half_precision {
                    Self::new_typed::<VectorElementTypeHalf>(
                        device,
                        GpuVectorStorageElementType::Float16,
                        vector_storage.distance(),
                        vector_storage.total_vector_count(),
                        |id| {
                            VectorElementTypeHalf::slice_from_float_cow(Cow::Borrowed(
                                vector_storage.get_dense(id),
                            ))
                        },
                        None,
                    )
                } else {
                    Self::new_typed::<VectorElementType>(
                        device,
                        GpuVectorStorageElementType::Float32,
                        vector_storage.distance(),
                        vector_storage.total_vector_count(),
                        |id| Cow::Borrowed(vector_storage.get_dense(id)),
                        None,
                    )
                }
            }
            VectorStorageEnum::DenseSimpleByte(vector_storage) => {
                Self::new_typed::<VectorElementTypeByte>(
                    device,
                    GpuVectorStorageElementType::Uint8,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| Cow::Borrowed(vector_storage.get_dense(id)),
                    None,
                )
            }
            VectorStorageEnum::DenseSimpleHalf(vector_storage) => {
                Self::new_typed::<VectorElementTypeHalf>(
                    device,
                    GpuVectorStorageElementType::Float16,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| Cow::Borrowed(vector_storage.get_dense(id)),
                    None,
                )
            }
            VectorStorageEnum::DenseMemmap(vector_storage) => {
                if force_half_precision {
                    Self::new_typed::<VectorElementTypeHalf>(
                        device,
                        GpuVectorStorageElementType::Float16,
                        vector_storage.distance(),
                        vector_storage.total_vector_count(),
                        |id| {
                            VectorElementTypeHalf::slice_from_float_cow(Cow::Borrowed(
                                vector_storage.get_dense(id),
                            ))
                        },
                        None,
                    )
                } else {
                    Self::new_typed::<VectorElementType>(
                        device,
                        GpuVectorStorageElementType::Float32,
                        vector_storage.distance(),
                        vector_storage.total_vector_count(),
                        |id| Cow::Borrowed(vector_storage.get_dense(id)),
                        None,
                    )
                }
            }
            VectorStorageEnum::DenseMemmapByte(vector_storage) => {
                Self::new_typed::<VectorElementTypeByte>(
                    device,
                    GpuVectorStorageElementType::Uint8,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| Cow::Borrowed(vector_storage.get_dense(id)),
                    None,
                )
            }
            VectorStorageEnum::DenseMemmapHalf(vector_storage) => {
                Self::new_typed::<VectorElementTypeHalf>(
                    device,
                    GpuVectorStorageElementType::Float16,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| Cow::Borrowed(vector_storage.get_dense(id)),
                    None,
                )
            }
            VectorStorageEnum::DenseAppendableMemmap(vector_storage) => {
                if force_half_precision {
                    Self::new_typed::<VectorElementTypeHalf>(
                        device,
                        GpuVectorStorageElementType::Float16,
                        vector_storage.distance(),
                        vector_storage.total_vector_count(),
                        |id| {
                            VectorElementTypeHalf::slice_from_float_cow(Cow::Borrowed(
                                vector_storage.get_dense(id),
                            ))
                        },
                        None,
                    )
                } else {
                    Self::new_typed::<VectorElementType>(
                        device,
                        GpuVectorStorageElementType::Float32,
                        vector_storage.distance(),
                        vector_storage.total_vector_count(),
                        |id| Cow::Borrowed(vector_storage.get_dense(id)),
                        None,
                    )
                }
            }
            VectorStorageEnum::DenseAppendableMemmapByte(vector_storage) => {
                Self::new_typed::<VectorElementTypeByte>(
                    device,
                    GpuVectorStorageElementType::Uint8,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| Cow::Borrowed(vector_storage.get_dense(id)),
                    None,
                )
            }
            VectorStorageEnum::DenseAppendableMemmapHalf(vector_storage) => {
                Self::new_typed::<VectorElementTypeHalf>(
                    device,
                    GpuVectorStorageElementType::Float16,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| Cow::Borrowed(vector_storage.get_dense(id)),
                    None,
                )
            }
            VectorStorageEnum::DenseAppendableInRam(vector_storage) => {
                if force_half_precision {
                    Self::new_typed::<VectorElementTypeHalf>(
                        device,
                        GpuVectorStorageElementType::Float16,
                        vector_storage.distance(),
                        vector_storage.total_vector_count(),
                        |id| {
                            VectorElementTypeHalf::slice_from_float_cow(Cow::Borrowed(
                                vector_storage.get_dense(id),
                            ))
                        },
                        None,
                    )
                } else {
                    Self::new_typed::<VectorElementType>(
                        device,
                        GpuVectorStorageElementType::Float32,
                        vector_storage.distance(),
                        vector_storage.total_vector_count(),
                        |id| Cow::Borrowed(vector_storage.get_dense(id)),
                        None,
                    )
                }
            }
            VectorStorageEnum::DenseAppendableInRamByte(vector_storage) => {
                Self::new_typed::<VectorElementTypeByte>(
                    device,
                    GpuVectorStorageElementType::Uint8,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| Cow::Borrowed(vector_storage.get_dense(id)),
                    None,
                )
            }
            VectorStorageEnum::DenseAppendableInRamHalf(vector_storage) => {
                Self::new_typed::<VectorElementTypeHalf>(
                    device,
                    GpuVectorStorageElementType::Float16,
                    vector_storage.distance(),
                    vector_storage.total_vector_count(),
                    |id| Cow::Borrowed(vector_storage.get_dense(id)),
                    None,
                )
            }
            VectorStorageEnum::SparseSimple(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            VectorStorageEnum::MultiDenseSimple(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            VectorStorageEnum::MultiDenseSimpleByte(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            VectorStorageEnum::MultiDenseSimpleHalf(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            VectorStorageEnum::MultiDenseAppendableMemmap(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            VectorStorageEnum::MultiDenseAppendableMemmapByte(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            VectorStorageEnum::MultiDenseAppendableMemmapHalf(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            VectorStorageEnum::MultiDenseAppendableInRam(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            VectorStorageEnum::MultiDenseAppendableInRamByte(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
            VectorStorageEnum::MultiDenseAppendableInRamHalf(_) => Err(OperationError::from(
                gpu::GpuError::NotSupported("Multivectors".to_string()),
            )),
        }
    }

    fn new_typed<'a, TElement: PrimitiveVectorElement>(
        device: Arc<gpu::Device>,
        element_type: GpuVectorStorageElementType,
        distance: Distance,
        count: usize,
        get_vector: impl Fn(PointOffsetType) -> Cow<'a, [TElement]>,
        quantization: Option<GpuQuantization>,
    ) -> OperationResult<Self> {
        let timer = std::time::Instant::now();

        let dim = get_vector(0).len();

        let capacity = Self::get_capacity(&device, dim);
        let upload_points_count = UPLOAD_CHUNK_SIZE / (capacity * std::mem::size_of::<TElement>());

        let points_in_storage_count = Self::get_points_in_storage_count(count);
        let vectors_buffer: Vec<Arc<gpu::Buffer>> = (0..STORAGES_COUNT)
            .map(|_| -> gpu::GpuResult<Arc<gpu::Buffer>> {
                Ok(gpu::Buffer::new(
                    device.clone(),
                    "Vector storage buffer",
                    gpu::BufferType::Storage,
                    points_in_storage_count * capacity * std::mem::size_of::<TElement>(),
                )?)
            })
            .collect::<gpu::GpuResult<Vec<_>>>()?;
        log::trace!("Storage buffer size {}", vectors_buffer[0].size());

        let mut upload_context = gpu::Context::new(device.clone())?;
        let staging_buffer = gpu::Buffer::new(
            device.clone(),
            "Vector storage upload staging buffer",
            gpu::BufferType::CpuToGpu,
            upload_points_count * capacity * std::mem::size_of::<TElement>(),
        )?;
        log::trace!(
            "Staging buffer size {}, upload_points_count = {}",
            staging_buffer.size(),
            upload_points_count
        );

        log::trace!("capacity = {}, count = {}", capacity, count);
        for (storage_index, vector_buffer) in vectors_buffer.iter().enumerate() {
            let mut gpu_offset = 0;
            let mut upload_size = 0;
            let mut upload_points = 0;
            let mut extended_vector = vec![TElement::default(); capacity];
            for point_id in 0..count {
                if point_id % STORAGES_COUNT != storage_index {
                    continue;
                }

                let vector = get_vector(point_id as PointOffsetType);
                extended_vector[..vector.len()].copy_from_slice(&vector);
                staging_buffer.upload_slice(
                    &extended_vector,
                    upload_points * capacity * std::mem::size_of::<TElement>(),
                )?;
                upload_size += capacity * std::mem::size_of::<TElement>();
                upload_points += 1;

                if upload_points == upload_points_count {
                    upload_context.copy_gpu_buffer(
                        staging_buffer.clone(),
                        vector_buffer.clone(),
                        0,
                        gpu_offset,
                        upload_size,
                    )?;
                    upload_context.run()?;
                    upload_context.wait_finish(GPU_TIMEOUT)?;

                    log::trace!(
                        "Uploaded {} vectors, {} MB",
                        upload_points,
                        upload_size / 1024 / 1024,
                    );

                    gpu_offset += upload_size;
                    upload_size = 0;
                    upload_points = 0;
                }
            }
            if upload_points > 0 {
                upload_context.copy_gpu_buffer(
                    staging_buffer.clone(),
                    vectors_buffer[storage_index].clone(),
                    0,
                    gpu_offset,
                    upload_size,
                )?;
                upload_context.run()?;
                upload_context.wait_finish(GPU_TIMEOUT)?;

                log::trace!(
                    "Uploaded {} vectors, {} MB",
                    upload_points,
                    upload_size / 1024 / 1024,
                );
            }
        }

        log::trace!(
            "Upload vector data to GPU time = {:?}, vector data size {} MB, element type: {:?}",
            timer.elapsed(),
            STORAGES_COUNT * points_in_storage_count * capacity * std::mem::size_of::<TElement>()
                / 1024
                / 1024,
            element_type,
        );

        let mut descriptor_set_layout_builder = gpu::DescriptorSetLayout::builder();
        for i in 0..STORAGES_COUNT {
            descriptor_set_layout_builder = descriptor_set_layout_builder.add_storage_buffer(i);
        }
        match &quantization {
            Some(GpuQuantization::Scalar(_)) => {
                descriptor_set_layout_builder =
                    descriptor_set_layout_builder.add_storage_buffer(STORAGES_COUNT);
            }
            Some(GpuQuantization::Product(_)) => {
                descriptor_set_layout_builder =
                    descriptor_set_layout_builder.add_storage_buffer(STORAGES_COUNT);
                descriptor_set_layout_builder =
                    descriptor_set_layout_builder.add_storage_buffer(STORAGES_COUNT + 1);
            }
            None => {}
        }
        let descriptor_set_layout = descriptor_set_layout_builder.build(device.clone())?;

        let mut descriptor_set_builder = gpu::DescriptorSet::builder(descriptor_set_layout.clone());
        for (i, vector_buffer) in vectors_buffer.iter().enumerate() {
            descriptor_set_builder =
                descriptor_set_builder.add_storage_buffer(i, vector_buffer.clone());
        }
        match &quantization {
            Some(GpuQuantization::Scalar(sq)) => {
                descriptor_set_builder = descriptor_set_builder
                    .add_storage_buffer(STORAGES_COUNT, sq.offsets_buffer.clone());
            }
            Some(GpuQuantization::Product(pq)) => {
                descriptor_set_builder = descriptor_set_builder
                    .add_storage_buffer(STORAGES_COUNT, pq.centroids_buffer.clone());
                descriptor_set_builder = descriptor_set_builder
                    .add_storage_buffer(STORAGES_COUNT + 1, pq.vector_division_buffer.clone());
            }
            None => {}
        }
        let descriptor_set = descriptor_set_builder.build()?;

        Ok(Self {
            device,
            vectors_buffer,
            descriptor_set_layout,
            descriptor_set,
            dim: capacity,
            count,
            element_type,
            distance,
            quantization,
        })
    }

    pub fn get_capacity(device: &Arc<gpu::Device>, dim: usize) -> usize {
        let alignment = device.subgroup_size() * ELEMENTS_PER_SUBGROUP;
        dim + (alignment - dim % alignment) % alignment
    }

    pub fn get_points_in_storage_count(num_vectors: usize) -> usize {
        (num_vectors + (STORAGES_COUNT - num_vectors % STORAGES_COUNT)) / STORAGES_COUNT
    }

    pub fn create_sq_offsets_buffer<TStorage: EncodedStorage>(
        device: Arc<gpu::Device>,
        quantized_storage: &EncodedVectorsU8<TStorage>,
        count: usize,
    ) -> OperationResult<Arc<gpu::Buffer>> {
        let sq_offsets_buffer = gpu::Buffer::new(
            device.clone(),
            "SQ offsets buffer",
            gpu::BufferType::Storage,
            count * std::mem::size_of::<f32>(),
        )?;

        let sq_offsets_staging_buffer = gpu::Buffer::new(
            device.clone(),
            "SQ offsets staging buffer",
            gpu::BufferType::CpuToGpu,
            sq_offsets_buffer.size(),
        )?;

        let mut upload_context = gpu::Context::new(device.clone())?;

        for i in 0..count {
            let (offset, _) = quantized_storage.get_quantized_vector(i as PointOffsetType);
            sq_offsets_staging_buffer.upload(&offset, i * std::mem::size_of::<f32>())?;
        }

        upload_context.copy_gpu_buffer(
            sq_offsets_staging_buffer,
            sq_offsets_buffer.clone(),
            0,
            0,
            sq_offsets_buffer.size(),
        )?;
        upload_context.run()?;
        upload_context.wait_finish(GPU_TIMEOUT)?;

        Ok(sq_offsets_buffer)
    }

    pub fn create_pq_quantization<TStorage: EncodedStorage>(
        device: Arc<gpu::Device>,
        quantized_storage: &EncodedVectorsPQ<TStorage>,
    ) -> OperationResult<GpuProductQuantization> {
        let centroids_buffer = gpu::Buffer::new(
            device.clone(),
            "PQ centroids buffer",
            gpu::BufferType::Storage,
            quantized_storage
                .get_metadata()
                .centroids
                .iter()
                .map(|c| c.len())
                .sum::<usize>()
                * std::mem::size_of::<f32>(),
        )?;
        let centroids_staging_buffer = gpu::Buffer::new(
            device.clone(),
            "PQ centroids staging buffer",
            gpu::BufferType::CpuToGpu,
            centroids_buffer.size(),
        )?;
        let vector_division_buffer = gpu::Buffer::new(
            device.clone(),
            "PQ vector division buffer",
            gpu::BufferType::Storage,
            quantized_storage.get_metadata().vector_division.len() * std::mem::size_of::<u32>() * 2,
        )?;
        let vector_division_staging_buffer = gpu::Buffer::new(
            device.clone(),
            "PQ vector division staging buffer",
            gpu::BufferType::CpuToGpu,
            vector_division_buffer.size(),
        )?;

        let mut upload_context = gpu::Context::new(device.clone())?;

        let mut centroids_offset = 0;
        for centroids in &quantized_storage.get_metadata().centroids {
            centroids_staging_buffer.upload(centroids, centroids_offset)?;
            centroids_offset += centroids.len() * std::mem::size_of::<f32>();
        }

        upload_context.copy_gpu_buffer(
            centroids_staging_buffer,
            centroids_buffer.clone(),
            0,
            0,
            centroids_buffer.size(),
        )?;

        let vector_division: Vec<_> = quantized_storage
            .get_metadata()
            .vector_division
            .iter()
            .flat_map(|range| [range.start as u32, range.end as u32].into_iter())
            .collect();
        vector_division_staging_buffer.upload(&vector_division, 0)?;

        upload_context.copy_gpu_buffer(
            vector_division_staging_buffer,
            vector_division_buffer.clone(),
            0,
            0,
            vector_division_buffer.size(),
        )?;

        upload_context.run()?;
        upload_context.wait_finish(GPU_TIMEOUT)?;

        Ok(GpuProductQuantization {
            centroids_buffer,
            vector_division_buffer,
            divisions_count: quantized_storage.get_metadata().vector_division.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use bitvec::vec::BitVec;
    use parking_lot::RwLock;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rocksdb::DB;

    use super::*;
    use crate::common::rocksdb_wrapper::{open_db, DB_VECTOR_CF};
    use crate::data_types::vectors::{QueryVector, VectorRef};
    use crate::fixtures::index_fixtures::random_vector;
    use crate::fixtures::payload_fixtures::random_dense_byte_vector;
    use crate::index::hnsw_index::gpu::shader_builder::ShaderBuilder;
    use crate::spaces::metric::Metric;
    use crate::spaces::simple::{CosineMetric, DotProductMetric, EuclidMetric, ManhattanMetric};
    use crate::types::{
        BinaryQuantization, BinaryQuantizationConfig, Distance, ProductQuantization,
        ProductQuantizationConfig, QuantizationConfig, ScalarQuantization,
        ScalarQuantizationConfig,
    };
    use crate::vector_storage::dense::simple_dense_vector_storage::{
        open_simple_dense_byte_vector_storage, open_simple_dense_half_vector_storage,
        open_simple_dense_vector_storage,
    };
    use crate::vector_storage::{new_raw_scorer, RawScorer};

    #[derive(Debug, Clone, Copy)]
    enum TestElementType {
        Float32,
        Float16,
        Uint8,
    }

    fn open_vector_storage(
        path: &Path,
        dim: usize,
        element_type: TestElementType,
    ) -> VectorStorageEnum {
        let db = open_db(path, &[DB_VECTOR_CF]).unwrap();

        match element_type {
            TestElementType::Float32 => open_simple_dense_vector_storage(
                db,
                DB_VECTOR_CF,
                dim,
                Distance::Dot,
                &false.into(),
            )
            .unwrap(),
            TestElementType::Float16 => open_simple_dense_half_vector_storage(
                db,
                DB_VECTOR_CF,
                dim,
                Distance::Dot,
                &false.into(),
            )
            .unwrap(),
            TestElementType::Uint8 => open_simple_dense_byte_vector_storage(
                db,
                DB_VECTOR_CF,
                dim,
                Distance::Dot,
                &false.into(),
            )
            .unwrap(),
        }
    }

    fn test_gpu_vector_storage_scoring_impl(
        element_type: TestElementType,
        force_half_precision: bool,
    ) -> GpuVectorStorageElementType {
        let num_vectors = 2048;
        let dim = 128;
        let test_point_id = 0usize;

        let mut rnd = StdRng::seed_from_u64(42);
        let points = (0..num_vectors)
            .map(|_| match element_type {
                TestElementType::Float32 => random_vector(&mut rnd, dim),
                TestElementType::Float16 => random_vector(&mut rnd, dim),
                TestElementType::Uint8 => random_dense_byte_vector(&mut rnd, dim),
            })
            .collect::<Vec<_>>();

        let dir = tempfile::Builder::new().prefix("db_dir").tempdir().unwrap();
        let mut storage = open_vector_storage(dir.path(), dim, element_type);
        points.iter().enumerate().for_each(|(i, vec)| {
            storage
                .insert_vector(i as PointOffsetType, vec.into())
                .unwrap();
        });

        let debug_messenger = gpu::PanicIfErrorMessenger {};
        let instance = gpu::Instance::new(Some(&debug_messenger), None, false).unwrap();
        let device = gpu::Device::new(instance.clone(), &instance.physical_devices()[0]).unwrap();

        let gpu_vector_storage =
            GpuVectorStorage::new(device.clone(), &storage, None, force_half_precision).unwrap();

        let scores_buffer = gpu::Buffer::new(
            device.clone(),
            "Scores buffer",
            gpu::BufferType::Storage,
            num_vectors * std::mem::size_of::<f32>(),
        )
        .unwrap();

        let descriptor_set_layout = gpu::DescriptorSetLayout::builder()
            .add_storage_buffer(0)
            .build(device.clone())
            .unwrap();

        let descriptor_set = gpu::DescriptorSet::builder(descriptor_set_layout.clone())
            .add_storage_buffer(0, scores_buffer.clone())
            .build()
            .unwrap();

        let shader = ShaderBuilder::new(device.clone())
            .with_shader_code(include_str!("shaders/tests/test_vector_storage.comp"))
            .with_gpu_vector_storage(&gpu_vector_storage)
            .build()
            .unwrap();

        let pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, gpu_vector_storage.descriptor_set_layout.clone())
            .add_shader(shader.clone())
            .build(device.clone())
            .unwrap();

        let mut context = gpu::Context::new(device.clone()).unwrap();
        context
            .bind_pipeline(
                pipeline,
                &[descriptor_set, gpu_vector_storage.descriptor_set.clone()],
            )
            .unwrap();
        context.dispatch(num_vectors, 1, 1).unwrap();

        let timer = std::time::Instant::now();
        context.run().unwrap();
        context.wait_finish(GPU_TIMEOUT).unwrap();
        log::trace!("GPU scoring time = {:?}", timer.elapsed());

        let staging_buffer = gpu::Buffer::new(
            device.clone(),
            "Scores staging buffer",
            gpu::BufferType::GpuToCpu,
            num_vectors * std::mem::size_of::<f32>(),
        )
        .unwrap();
        context
            .copy_gpu_buffer(
                scores_buffer,
                staging_buffer.clone(),
                0,
                0,
                num_vectors * std::mem::size_of::<f32>(),
            )
            .unwrap();
        context.run().unwrap();
        context.wait_finish(GPU_TIMEOUT).unwrap();

        let mut scores = vec![0.0f32; num_vectors];
        staging_buffer.download_slice(&mut scores, 0).unwrap();

        let timer = std::time::Instant::now();
        for i in 0..num_vectors {
            let score = DotProductMetric::similarity(&points[test_point_id], &points[i]);
            assert!((score - scores[i]).abs() < 0.01);
        }
        log::trace!("CPU scoring time = {:?}", timer.elapsed());

        gpu_vector_storage.element_type
    }

    #[test]
    fn test_gpu_vector_storage_scoring() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init();
        let element = test_gpu_vector_storage_scoring_impl(TestElementType::Float32, false);
        assert_eq!(element, GpuVectorStorageElementType::Float32);
    }

    #[test]
    fn test_gpu_vector_storage_scoring_f16() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init();
        let element = test_gpu_vector_storage_scoring_impl(TestElementType::Float16, false);
        assert_eq!(element, GpuVectorStorageElementType::Float16);
    }

    #[test]
    fn test_gpu_vector_storage_scoring_u8() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init();
        let element = test_gpu_vector_storage_scoring_impl(TestElementType::Uint8, false);
        assert_eq!(element, GpuVectorStorageElementType::Uint8);
    }

    #[test]
    fn test_gpu_vector_storage_force_half_precision() {
        let element = test_gpu_vector_storage_scoring_impl(TestElementType::Float32, true);
        assert_eq!(element, GpuVectorStorageElementType::Float16);
    }

    #[test]
    fn test_gpu_vector_storage_binary_quantization() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init();

        let num_vectors = 16;
        let dim = 1024;
        let test_point_id = 0usize;

        let mut rnd = StdRng::seed_from_u64(42);
        let points = (0..num_vectors)
            .map(|_| random_vector(&mut rnd, dim))
            .collect::<Vec<_>>();

        let dir = tempfile::Builder::new().prefix("db_dir").tempdir().unwrap();
        let db = open_db(dir.path(), &[DB_VECTOR_CF]).unwrap();
        let mut storage =
            open_simple_dense_vector_storage(db, DB_VECTOR_CF, dim, Distance::Dot, &false.into())
                .unwrap();

        points.iter().enumerate().for_each(|(i, vec)| {
            storage
                .insert_vector(i as PointOffsetType, vec.into())
                .unwrap();
        });

        let quantized_vectors = QuantizedVectors::create(
            &storage,
            &QuantizationConfig::Binary(BinaryQuantization {
                binary: BinaryQuantizationConfig {
                    always_ram: Some(true),
                },
            }),
            dir.path(),
            1,
            &false.into(),
        )
        .unwrap();

        let debug_messenger = gpu::PanicIfErrorMessenger {};
        let instance = gpu::Instance::new(Some(&debug_messenger), None, false).unwrap();
        let device = gpu::Device::new(instance.clone(), &instance.physical_devices()[0]).unwrap();

        let gpu_vector_storage =
            GpuVectorStorage::new(device.clone(), &storage, Some(&quantized_vectors), false)
                .unwrap();
        assert_eq!(
            gpu_vector_storage.element_type,
            GpuVectorStorageElementType::Binary
        );

        let scores_buffer = gpu::Buffer::new(
            device.clone(),
            "Scores buffer",
            gpu::BufferType::Storage,
            num_vectors * std::mem::size_of::<f32>(),
        )
        .unwrap();

        let descriptor_set_layout = gpu::DescriptorSetLayout::builder()
            .add_storage_buffer(0)
            .build(device.clone())
            .unwrap();

        let descriptor_set = gpu::DescriptorSet::builder(descriptor_set_layout.clone())
            .add_storage_buffer(0, scores_buffer.clone())
            .build()
            .unwrap();

        let shader = ShaderBuilder::new(device.clone())
            .with_shader_code(include_str!("shaders/tests/test_vector_storage.comp"))
            .with_gpu_vector_storage(&gpu_vector_storage)
            .build()
            .unwrap();

        let pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, gpu_vector_storage.descriptor_set_layout.clone())
            .add_shader(shader.clone())
            .build(device.clone())
            .unwrap();

        let mut context = gpu::Context::new(device.clone()).unwrap();
        context
            .bind_pipeline(
                pipeline,
                &[descriptor_set, gpu_vector_storage.descriptor_set.clone()],
            )
            .unwrap();
        context.dispatch(num_vectors, 1, 1).unwrap();

        let timer = std::time::Instant::now();
        context.run().unwrap();
        context.wait_finish(GPU_TIMEOUT).unwrap();
        log::trace!("GPU scoring time = {:?}", timer.elapsed());

        let staging_buffer = gpu::Buffer::new(
            device.clone(),
            "Scores staging buffer",
            gpu::BufferType::GpuToCpu,
            num_vectors * std::mem::size_of::<f32>(),
        )
        .unwrap();
        context
            .copy_gpu_buffer(
                scores_buffer,
                staging_buffer.clone(),
                0,
                0,
                num_vectors * std::mem::size_of::<f32>(),
            )
            .unwrap();
        context.run().unwrap();
        context.wait_finish(GPU_TIMEOUT).unwrap();

        let mut scores = vec![0.0f32; num_vectors];
        staging_buffer.download_slice(&mut scores, 0).unwrap();

        let stopped = false.into();
        let point_deleted = BitVec::repeat(false, num_vectors);
        let scorer = quantized_vectors
            .raw_scorer(
                points[test_point_id].clone().into(),
                &point_deleted,
                &point_deleted,
                &stopped,
            )
            .unwrap();
        for i in 0..num_vectors {
            let score =
                scorer.score_internal(test_point_id as PointOffsetType, i as PointOffsetType);
            assert_eq!(score, scores[i]);
        }
    }

    #[test]
    fn test_gpu_vector_storage_sq_old() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init();

        let num_vectors = 16;
        let dim = 1024;
        let test_point_id = 0usize;

        let mut rnd = StdRng::seed_from_u64(42);
        let points = (0..num_vectors)
            .map(|_| random_vector(&mut rnd, dim))
            .collect::<Vec<_>>();

        let dir = tempfile::Builder::new().prefix("db_dir").tempdir().unwrap();
        let db = open_db(dir.path(), &[DB_VECTOR_CF]).unwrap();
        let mut storage =
            open_simple_dense_vector_storage(db, DB_VECTOR_CF, dim, Distance::Dot, &false.into())
                .unwrap();

        points.iter().enumerate().for_each(|(i, vec)| {
            storage
                .insert_vector(i as PointOffsetType, vec.into())
                .unwrap();
        });

        let quantized_vectors = QuantizedVectors::create(
            &storage,
            &QuantizationConfig::Scalar(ScalarQuantization {
                scalar: ScalarQuantizationConfig {
                    always_ram: Some(true),
                    r#type: crate::types::ScalarType::Int8,
                    quantile: Some(0.99),
                },
            }),
            dir.path(),
            1,
            &false.into(),
        )
        .unwrap();

        let debug_messenger = gpu::PanicIfErrorMessenger {};
        let instance = gpu::Instance::new(Some(&debug_messenger), None, false).unwrap();
        let device = gpu::Device::new(instance.clone(), &instance.physical_devices()[0]).unwrap();

        let gpu_vector_storage =
            GpuVectorStorage::new(device.clone(), &storage, Some(&quantized_vectors), false)
                .unwrap();
        assert_eq!(
            gpu_vector_storage.element_type,
            GpuVectorStorageElementType::SQ
        );

        let scores_buffer = gpu::Buffer::new(
            device.clone(),
            "Scores buffer",
            gpu::BufferType::Storage,
            num_vectors * std::mem::size_of::<f32>(),
        )
        .unwrap();

        let descriptor_set_layout = gpu::DescriptorSetLayout::builder()
            .add_storage_buffer(0)
            .build(device.clone())
            .unwrap();

        let descriptor_set = gpu::DescriptorSet::builder(descriptor_set_layout.clone())
            .add_storage_buffer(0, scores_buffer.clone())
            .build()
            .unwrap();

        let shader = ShaderBuilder::new(device.clone())
            .with_shader_code(include_str!("shaders/tests/test_vector_storage.comp"))
            .with_gpu_vector_storage(&gpu_vector_storage)
            .build()
            .unwrap();

        let pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, gpu_vector_storage.descriptor_set_layout.clone())
            .add_shader(shader.clone())
            .build(device.clone())
            .unwrap();

        let mut context = gpu::Context::new(device.clone()).unwrap();
        context
            .bind_pipeline(
                pipeline,
                &[descriptor_set, gpu_vector_storage.descriptor_set.clone()],
            )
            .unwrap();
        context.dispatch(num_vectors, 1, 1).unwrap();

        let timer = std::time::Instant::now();
        context.run().unwrap();
        context.wait_finish(GPU_TIMEOUT).unwrap();
        log::trace!("GPU scoring time = {:?}", timer.elapsed());

        let staging_buffer = gpu::Buffer::new(
            device.clone(),
            "Scores staging buffer",
            gpu::BufferType::GpuToCpu,
            num_vectors * std::mem::size_of::<f32>(),
        )
        .unwrap();
        context
            .copy_gpu_buffer(
                scores_buffer,
                staging_buffer.clone(),
                0,
                0,
                num_vectors * std::mem::size_of::<f32>(),
            )
            .unwrap();
        context.run().unwrap();
        context.wait_finish(GPU_TIMEOUT).unwrap();

        let mut scores = vec![0.0f32; num_vectors];
        staging_buffer.download_slice(&mut scores, 0).unwrap();

        let stopped = false.into();
        let point_deleted = BitVec::repeat(false, num_vectors);
        let scorer = quantized_vectors
            .raw_scorer(
                points[test_point_id].clone().into(),
                &point_deleted,
                &point_deleted,
                &stopped,
            )
            .unwrap();
        for i in 0..num_vectors {
            let score =
                scorer.score_internal(test_point_id as PointOffsetType, i as PointOffsetType);
            assert!((score - scores[i]).abs() < 0.01);
        }
    }

    static DISTANCES: [Distance; 4] = [
        Distance::Cosine,
        Distance::Dot,
        Distance::Euclid,
        Distance::Manhattan,
    ];
    static ELEMENT_TYPES: [TestElementType; 3] = [
        TestElementType::Float32,
        TestElementType::Float16,
        TestElementType::Uint8,
    ];
    static DIMS: [usize; 3] = [16, 512, 256 + 17];
    static NUM_VECTORS: usize = 2048 + 17;

    #[test]
    fn test_gpu_vector_storage_sq() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init();

        let quantization_config = QuantizationConfig::Scalar(ScalarQuantization {
            scalar: ScalarQuantizationConfig {
                always_ram: Some(true),
                r#type: crate::types::ScalarType::Int8,
                quantile: Some(0.99),
            },
        });

        for (&distance, &element_type, &dim) in
            itertools::iproduct!(DISTANCES.iter(), ELEMENT_TYPES.iter(), DIMS.iter())
        {
            let precision = get_precision(element_type, dim, distance);
            log::info!(
                "Testing SQ distance {:?}, element type {:?}, dim {} with precision {}",
                distance,
                element_type,
                dim,
                precision
            );
            test_gpu_vector_storage_impl(
                element_type,
                NUM_VECTORS,
                dim,
                distance,
                Some(quantization_config.clone()),
                false,
                10.0 * precision,
            );
        }
    }

    #[test]
    fn test_gpu_vector_storage_bq() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init();

        let quantization_config = QuantizationConfig::Binary(BinaryQuantization {
            binary: BinaryQuantizationConfig {
                always_ram: Some(true),
            },
        });

        for (&distance, &element_type, &dim) in
            itertools::iproduct!(DISTANCES.iter(), ELEMENT_TYPES.iter(), DIMS.iter())
        {
            let precision = get_precision(element_type, dim, distance);
            log::info!(
                "Testing BQ distance {:?}, element type {:?}, dim {} with precision {}",
                distance,
                element_type,
                dim,
                precision
            );
            test_gpu_vector_storage_impl(
                element_type,
                NUM_VECTORS,
                dim,
                distance,
                Some(quantization_config.clone()),
                false,
                10.0 * precision,
            );
        }
    }

    #[test]
    fn test_gpu_vector_storage_pq() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init();

        let quantization_config = QuantizationConfig::Product(ProductQuantization {
            product: ProductQuantizationConfig {
                always_ram: Some(true),
                compression: crate::types::CompressionRatio::X4,
            },
        });

        for (&distance, &element_type, &dim) in
            itertools::iproduct!(DISTANCES.iter(), ELEMENT_TYPES.iter(), DIMS.iter())
        {
            let precision = get_precision(element_type, dim, distance);
            log::info!(
                "Testing PQ distance {:?}, element type {:?}, dim {} with precision {}",
                distance,
                element_type,
                dim,
                precision
            );
            test_gpu_vector_storage_impl(
                element_type,
                NUM_VECTORS,
                dim,
                distance,
                Some(quantization_config.clone()),
                false,
                10.0 * precision,
            );
        }
    }

    #[test]
    fn test_gpu_vector_storage() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init();

        for (&distance, &element_type, &dim) in
            itertools::iproduct!(DISTANCES.iter(), ELEMENT_TYPES.iter(), DIMS.iter())
        {
            let precision = get_precision(element_type, dim, distance);
            log::info!(
                "Testing distance {:?}, element type {:?}, dim {} with precision {}",
                distance,
                element_type,
                dim,
                precision
            );
            test_gpu_vector_storage_impl(
                element_type,
                NUM_VECTORS,
                dim,
                distance,
                None,
                false,
                precision,
            );
        }
    }

    fn get_precision(element_type: TestElementType, dim: usize, distance: Distance) -> f32 {
        let distance_persision = match distance {
            Distance::Cosine => 0.01,
            Distance::Dot => 0.01,
            Distance::Euclid => dim as f32 * 0.0001,
            Distance::Manhattan => dim as f32 * 0.0001,
        };
        match element_type {
            TestElementType::Float32 => distance_persision,
            TestElementType::Float16 => distance_persision * 5.0,
            TestElementType::Uint8 => distance_persision * 10.0,
        }
    }

    fn create_vector_storage(
        path: &Path,
        element_type: TestElementType,
        num_vectors: usize,
        dim: usize,
        distance: Distance,
    ) -> VectorStorageEnum {
        let db = open_db(path, &[DB_VECTOR_CF]).unwrap();
        match element_type {
            TestElementType::Float32 => create_vector_storage_f32(db, num_vectors, dim, distance),
            TestElementType::Float16 => create_vector_storage_f16(db, num_vectors, dim, distance),
            TestElementType::Uint8 => create_vector_storage_u8(db, num_vectors, dim, distance),
        }
    }

    fn create_vector_storage_f32(
        db: Arc<RwLock<DB>>,
        num_vectors: usize,
        dim: usize,
        distance: Distance,
    ) -> VectorStorageEnum {
        let mut rnd = StdRng::seed_from_u64(42);
        let mut vector_storage =
            open_simple_dense_vector_storage(db, DB_VECTOR_CF, dim, distance, &false.into())
                .unwrap();
        for i in 0..num_vectors {
            let vec = random_vector(&mut rnd, dim);
            let vec = match distance {
                Distance::Cosine => <CosineMetric as Metric<VectorElementType>>::preprocess(vec),
                Distance::Euclid => <EuclidMetric as Metric<VectorElementType>>::preprocess(vec),
                Distance::Dot => <DotProductMetric as Metric<VectorElementType>>::preprocess(vec),
                Distance::Manhattan => {
                    <ManhattanMetric as Metric<VectorElementType>>::preprocess(vec)
                }
            };
            let vec_ref = VectorRef::from(&vec);
            vector_storage
                .insert_vector(i as PointOffsetType, vec_ref)
                .unwrap();
        }
        vector_storage
    }

    fn create_vector_storage_f16(
        db: Arc<RwLock<DB>>,
        num_vectors: usize,
        dim: usize,
        distance: Distance,
    ) -> VectorStorageEnum {
        let mut rnd = StdRng::seed_from_u64(42);
        let mut vector_storage =
            open_simple_dense_half_vector_storage(db, DB_VECTOR_CF, dim, distance, &false.into())
                .unwrap();
        for i in 0..num_vectors {
            let vec = random_vector(&mut rnd, dim);
            let vec = match distance {
                Distance::Cosine => {
                    <CosineMetric as Metric<VectorElementTypeHalf>>::preprocess(vec)
                }
                Distance::Euclid => {
                    <EuclidMetric as Metric<VectorElementTypeHalf>>::preprocess(vec)
                }
                Distance::Dot => {
                    <DotProductMetric as Metric<VectorElementTypeHalf>>::preprocess(vec)
                }
                Distance::Manhattan => {
                    <ManhattanMetric as Metric<VectorElementTypeHalf>>::preprocess(vec)
                }
            };
            let vec_ref = VectorRef::from(&vec);
            vector_storage
                .insert_vector(i as PointOffsetType, vec_ref)
                .unwrap();
        }
        vector_storage
    }

    fn create_vector_storage_u8(
        db: Arc<RwLock<DB>>,
        num_vectors: usize,
        dim: usize,
        distance: Distance,
    ) -> VectorStorageEnum {
        let mut rnd = StdRng::seed_from_u64(42);
        let mut vector_storage =
            open_simple_dense_byte_vector_storage(db, DB_VECTOR_CF, dim, distance, &false.into())
                .unwrap();
        for i in 0..num_vectors {
            let vec = random_dense_byte_vector(&mut rnd, dim);
            let vec = match distance {
                Distance::Cosine => {
                    <CosineMetric as Metric<VectorElementTypeByte>>::preprocess(vec)
                }
                Distance::Euclid => {
                    <EuclidMetric as Metric<VectorElementTypeByte>>::preprocess(vec)
                }
                Distance::Dot => {
                    <DotProductMetric as Metric<VectorElementTypeByte>>::preprocess(vec)
                }
                Distance::Manhattan => {
                    <ManhattanMetric as Metric<VectorElementTypeByte>>::preprocess(vec)
                }
            };
            let vec_ref = VectorRef::from(&vec);
            vector_storage
                .insert_vector(i as PointOffsetType, vec_ref)
                .unwrap();
        }
        vector_storage
    }

    fn test_gpu_vector_storage_impl(
        element_type: TestElementType,
        num_vectors: usize,
        dim: usize,
        distance: Distance,
        quantization_config: Option<QuantizationConfig>,
        force_half_precision: bool,
        precision: f32,
    ) {
        let test_point_id: PointOffsetType = 0;

        let dir = tempfile::Builder::new().prefix("db_dir").tempdir().unwrap();
        let storage = create_vector_storage(dir.path(), element_type, num_vectors, dim, distance);

        let quantized_vectors = if let Some(quantization_config) = &quantization_config {
            Some(
                QuantizedVectors::create(
                    &storage,
                    quantization_config,
                    dir.path(),
                    1,
                    &false.into(),
                )
                .unwrap(),
            )
        } else {
            None
        };

        let debug_messenger = gpu::PanicIfErrorMessenger {};
        let instance = gpu::Instance::new(Some(&debug_messenger), None, false).unwrap();
        let device = gpu::Device::new(instance.clone(), &instance.physical_devices()[0]).unwrap();

        let gpu_vector_storage = GpuVectorStorage::new(
            device.clone(),
            &storage,
            quantized_vectors.as_ref(),
            force_half_precision,
        )
        .unwrap();

        assert_eq!(
            gpu_vector_storage.element_type,
            if let Some(quantization_config) = quantization_config.as_ref() {
                match quantization_config {
                    QuantizationConfig::Scalar(_) => GpuVectorStorageElementType::SQ,
                    QuantizationConfig::Product(_) => GpuVectorStorageElementType::PQ,
                    QuantizationConfig::Binary(_) => GpuVectorStorageElementType::Binary,
                }
            } else {
                match element_type {
                    TestElementType::Float32 => GpuVectorStorageElementType::Float32,
                    TestElementType::Float16 => GpuVectorStorageElementType::Float16,
                    TestElementType::Uint8 => GpuVectorStorageElementType::Uint8,
                }
            }
        );

        let scores_buffer = gpu::Buffer::new(
            device.clone(),
            "Scores buffer",
            gpu::BufferType::Storage,
            num_vectors * std::mem::size_of::<f32>(),
        )
        .unwrap();

        let descriptor_set_layout = gpu::DescriptorSetLayout::builder()
            .add_storage_buffer(0)
            .build(device.clone())
            .unwrap();

        let descriptor_set = gpu::DescriptorSet::builder(descriptor_set_layout.clone())
            .add_storage_buffer(0, scores_buffer.clone())
            .build()
            .unwrap();

        let shader = ShaderBuilder::new(device.clone())
            .with_shader_code(include_str!("shaders/tests/test_vector_storage.comp"))
            .with_gpu_vector_storage(&gpu_vector_storage)
            .build()
            .unwrap();

        let pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, gpu_vector_storage.descriptor_set_layout.clone())
            .add_shader(shader.clone())
            .build(device.clone())
            .unwrap();

        let mut context = gpu::Context::new(device.clone()).unwrap();
        context
            .bind_pipeline(
                pipeline,
                &[descriptor_set, gpu_vector_storage.descriptor_set.clone()],
            )
            .unwrap();
        context.dispatch(num_vectors, 1, 1).unwrap();

        let timer = std::time::Instant::now();
        context.run().unwrap();
        context.wait_finish(GPU_TIMEOUT).unwrap();
        log::trace!("GPU scoring time = {:?}", timer.elapsed());

        let staging_buffer = gpu::Buffer::new(
            device.clone(),
            "Scores staging buffer",
            gpu::BufferType::GpuToCpu,
            num_vectors * std::mem::size_of::<f32>(),
        )
        .unwrap();
        context
            .copy_gpu_buffer(
                scores_buffer,
                staging_buffer.clone(),
                0,
                0,
                num_vectors * std::mem::size_of::<f32>(),
            )
            .unwrap();
        context.run().unwrap();
        context.wait_finish(GPU_TIMEOUT).unwrap();

        let mut scores = vec![0.0f32; num_vectors];
        staging_buffer.download_slice(&mut scores, 0).unwrap();

        let stopped = false.into();
        let point_deleted = BitVec::repeat(false, num_vectors);
        let query = QueryVector::Nearest(storage.get_vector(test_point_id).to_owned());
        let scorer: Box<dyn RawScorer> = if let Some(quantized_vectors) = quantized_vectors.as_ref()
        {
            quantized_vectors
                .raw_scorer(query, &point_deleted, &point_deleted, &stopped)
                .unwrap()
        } else {
            new_raw_scorer(query, &storage, &point_deleted).unwrap()
        };

        for i in 0..num_vectors {
            let score =
                scorer.score_internal(test_point_id as PointOffsetType, i as PointOffsetType);
            assert!((score - scores[i]).abs() < precision);
        }
    }
}
