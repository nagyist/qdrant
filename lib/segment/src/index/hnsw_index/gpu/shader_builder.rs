use std::collections::HashMap;
use std::sync::Arc;

use super::gpu_vector_storage::{GpuVectorStorage, GpuVectorStorageElementType};
use crate::common::operation_error::OperationResult;
use crate::index::hnsw_index::gpu::gpu_vector_storage::GpuQuantization;
use crate::types::Distance;

pub struct ShaderBuilder<'a> {
    device: Arc<gpu::Device>,
    shader_code: String,
    exact: Option<bool>,
    nearest_heap_ef: Option<usize>,
    nearest_heap_capacity: Option<usize>,
    candidates_heap_capacity: Option<usize>,
    links_capacity: Option<usize>,
    visited_flags_capacity: Option<usize>,
    shaders_map: HashMap<String, String>,
    gpu_vector_storage: Option<&'a GpuVectorStorage>,
}

impl<'a> ShaderBuilder<'a> {
    pub fn new(device: Arc<gpu::Device>) -> Self {
        let shaders_map = HashMap::from([
            (
                "bheap.comp".to_string(),
                include_str!("shaders/bheap.comp").to_string(),
            ),
            (
                "candidates_heap.comp".to_string(),
                include_str!("shaders/candidates_heap.comp").to_string(),
            ),
            (
                "common.comp".to_string(),
                include_str!("shaders/common.comp").to_string(),
            ),
            (
                "extensions.comp".to_string(),
                include_str!("shaders/extensions.comp").to_string(),
            ),
            (
                "iterators.comp".to_string(),
                include_str!("shaders/iterators.comp").to_string(),
            ),
            (
                "links.comp".to_string(),
                include_str!("shaders/links.comp").to_string(),
            ),
            (
                "nearest_heap.comp".to_string(),
                include_str!("shaders/nearest_heap.comp").to_string(),
            ),
            (
                "run_get_patch.comp".to_string(),
                include_str!("shaders/run_get_patch.comp").to_string(),
            ),
            (
                "run_greedy_search.comp".to_string(),
                include_str!("shaders/run_greedy_search.comp").to_string(),
            ),
            (
                "run_insert_vector.comp".to_string(),
                include_str!("shaders/run_insert_vector.comp").to_string(),
            ),
            (
                "search_context.comp".to_string(),
                include_str!("shaders/search_context.comp").to_string(),
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
            (
                "visited_flags.comp".to_string(),
                include_str!("shaders/visited_flags.comp").to_string(),
            ),
        ]);

        Self {
            device,
            shader_code: Default::default(),
            exact: None,
            nearest_heap_ef: None,
            nearest_heap_capacity: None,
            candidates_heap_capacity: None,
            links_capacity: None,
            visited_flags_capacity: None,
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

    pub fn with_exact(&mut self, exact: bool) -> &mut Self {
        self.exact = Some(exact);
        self
    }

    pub fn with_nearest_heap_ef(&mut self, nearest_heap_ef: usize) -> &mut Self {
        self.nearest_heap_ef = Some(nearest_heap_ef);
        self
    }

    pub fn with_nearest_heap_capacity(&mut self, nearest_heap_capacity: usize) -> &mut Self {
        self.nearest_heap_capacity = Some(nearest_heap_capacity);
        self
    }

    pub fn with_candidates_heap_capacity(&mut self, candidates_heap_capacity: usize) -> &mut Self {
        self.candidates_heap_capacity = Some(candidates_heap_capacity);
        self
    }

    pub fn with_links_capacity(&mut self, links_capacity: usize) -> &mut Self {
        self.links_capacity = Some(links_capacity);
        self
    }

    pub fn with_visited_flags_capacity(&mut self, visited_flags_capacity: usize) -> &mut Self {
        self.visited_flags_capacity = Some(visited_flags_capacity);
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

            // options.add_macro_definition("STORAGES_COUNT", Some(&gpu_vector_storage.storages_count.to_string()));
            // options.add_macro_definition("STORAGE_SIZE", Some(&gpu_vector_storage.storage_size.to_string()));
        }

        if self.exact == Some(true) {
            defines.insert("EXACT".to_owned(), None);
        }

        if let Some(nearest_heap_ef) = self.nearest_heap_ef {
            defines.insert(
                "NEAREST_HEAP_EF".to_owned(),
                Some(nearest_heap_ef.to_string()),
            );
        }

        if let Some(nearest_heap_capacity) = self.nearest_heap_capacity {
            defines.insert(
                "NEAREST_HEAP_CAPACITY".to_owned(),
                Some(nearest_heap_capacity.to_string()),
            );
        }

        if let Some(candidates_heap_capacity) = self.candidates_heap_capacity {
            defines.insert(
                "CANDIDATES_HEAP_CAPACITY".to_owned(),
                Some(candidates_heap_capacity.to_string()),
            );
        }

        if let Some(links_capacity) = self.links_capacity {
            defines.insert(
                "LINKS_CAPACITY".to_owned(),
                Some(links_capacity.to_string()),
            );
        }

        if let Some(visited_flags_capacity) = self.visited_flags_capacity {
            defines.insert(
                "VISITED_FLAGS_CAPACITY".to_owned(),
                Some(visited_flags_capacity.to_string()),
            );
        }

        let timer = std::time::Instant::now();
        let compiled = self
            .device
            .instance()
            .compile_shader(&self.shader_code, Some(&defines), Some(&self.shaders_map))
            .unwrap();
        log::debug!("Shader compilation took: {:?}", timer.elapsed());
        Ok(gpu::Shader::new(self.device.clone(), &compiled)?)
    }
}
