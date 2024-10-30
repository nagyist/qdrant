use std::collections::HashMap;
use std::sync::Arc;

use super::gpu_vector_storage::{GpuQuantization, GpuVectorStorage, GpuVectorStorageElementType};
use crate::common::operation_error::OperationResult;
use crate::types::Distance;

pub struct ShaderBuilder<'a> {
    device: Arc<gpu::Device>,
    shader_code: String,
    shaders_map: HashMap<String, String>,
    gpu_vector_storage: Option<&'a GpuVectorStorage>,
}

impl<'a> ShaderBuilder<'a> {
    pub fn new(device: Arc<gpu::Device>) -> Self {
        let shaders_map = HashMap::from([
            (
                "common.comp".to_string(),
                include_str!("shaders/common.comp").to_string(),
            ),
            (
                "extensions.comp".to_string(),
                include_str!("shaders/extensions.comp").to_string(),
            ),
            (
                "vector_storage.comp".to_string(),
                include_str!("shaders/vector_storage.comp").to_string(),
            ),
            (
                "vector_storage_bq.comp".to_string(),
                include_str!("shaders/vector_storage_bq.comp").to_string(),
            ),
            (
                "vector_storage_f16.comp".to_string(),
                include_str!("shaders/vector_storage_f16.comp").to_string(),
            ),
            (
                "vector_storage_f32.comp".to_string(),
                include_str!("shaders/vector_storage_f32.comp").to_string(),
            ),
            (
                "vector_storage_pq.comp".to_string(),
                include_str!("shaders/vector_storage_pq.comp").to_string(),
            ),
            (
                "vector_storage_sq.comp".to_string(),
                include_str!("shaders/vector_storage_sq.comp").to_string(),
            ),
            (
                "vector_storage_u8.comp".to_string(),
                include_str!("shaders/vector_storage_u8.comp").to_string(),
            ),
        ]);

        Self {
            device,
            shader_code: Default::default(),
            gpu_vector_storage: None,
            shaders_map,
        }
    }

    pub fn with_shader_code(&mut self, shader_code: &str) -> &mut Self {
        self.shader_code.push_str("\n");
        self.shader_code.push_str(shader_code);
        self
    }

    pub fn with_gpu_vector_storage(
        &mut self,
        gpu_vector_storage: &'a GpuVectorStorage,
    ) -> &mut Self {
        self.gpu_vector_storage = Some(gpu_vector_storage);
        self
    }

    pub fn build(&self) -> OperationResult<Arc<gpu::Shader>> {
        let mut defines = HashMap::new();

        defines.insert(
            "SUBGROUP_SIZE".to_owned(),
            Some(self.device.subgroup_size().to_string()),
        );

        if let Some(gpu_vector_storage) = self.gpu_vector_storage {
            match gpu_vector_storage.element_type {
                GpuVectorStorageElementType::Float32 => {
                    defines.insert("VECTOR_STORAGE_ELEMENT_FLOAT32".to_owned(), None);
                }
                GpuVectorStorageElementType::Float16 => {
                    defines.insert("VECTOR_STORAGE_ELEMENT_FLOAT16".to_owned(), None);
                }
                GpuVectorStorageElementType::Uint8 => {
                    defines.insert("VECTOR_STORAGE_ELEMENT_UINT8".to_owned(), None);
                }
                GpuVectorStorageElementType::Binary => {
                    defines.insert("VECTOR_STORAGE_ELEMENT_BINARY".to_owned(), None);
                }
                GpuVectorStorageElementType::SQ => {
                    defines.insert("VECTOR_STORAGE_ELEMENT_SQ".to_owned(), None);
                }
                GpuVectorStorageElementType::PQ => {
                    defines.insert("VECTOR_STORAGE_ELEMENT_PQ".to_owned(), None);
                }
            }

            match gpu_vector_storage.distance {
                Distance::Cosine => {
                    defines.insert("COSINE_DISTANCE".to_owned(), None);
                }
                Distance::Euclid => {
                    defines.insert("EUCLID_DISTANCE".to_owned(), None);
                }
                Distance::Dot => {
                    defines.insert("DOT_DISTANCE".to_owned(), None);
                }
                Distance::Manhattan => {
                    defines.insert("MANHATTAN_DISTANCE".to_owned(), None);
                }
            }

            match &gpu_vector_storage.quantization {
                Some(GpuQuantization::Binary(binary)) => {
                    defines.insert(
                        "BQ_SKIP_COUNT".to_owned(),
                        Some(binary.skip_count.to_string()),
                    );
                }
                Some(GpuQuantization::Scalar(scalar)) => {
                    defines.insert(
                        "SQ_MULTIPLIER".to_owned(),
                        Some(scalar.multiplier.to_string()),
                    );
                    defines.insert("SQ_DIFF".to_owned(), Some(scalar.diff.to_string()));
                }
                Some(GpuQuantization::Product(product)) => {
                    defines.insert(
                        "PQ_DIVISIONS_COUNT".to_owned(),
                        Some(product.divisions_count.to_string()),
                    );
                }
                None => {}
            }

            defines.insert("DIM".to_owned(), Some(gpu_vector_storage.dim.to_string()));
        }

        let timer = std::time::Instant::now();
        let compiled = self
            .device
            .instance()
            .compile_shader(&self.shader_code, Some(&defines), Some(&self.shaders_map))
            .unwrap();
        log::trace!("Shader compilation took: {:?}", timer.elapsed());
        Ok(gpu::Shader::new(self.device.clone(), &compiled)?)
    }
}
