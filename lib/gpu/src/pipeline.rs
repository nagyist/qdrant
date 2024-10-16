use std::sync::Arc;

use ash::vk;

use crate::*;

pub struct Pipeline {
    pub(crate) device: Arc<Device>,
    // keep a reference to the shader to prevent it from being dropped
    pub(crate) _shader: Arc<Shader>,
    pub(crate) descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>>,
    pub(crate) vk_pipeline_layout: vk::PipelineLayout,
    pub(crate) vk_pipeline: vk::Pipeline,
}

impl Resource for Pipeline {}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if self.vk_pipeline != vk::Pipeline::null() {
            unsafe {
                self.device
                    .vk_device
                    .destroy_pipeline(self.vk_pipeline, self.device.allocation_callbacks());
            }
            self.vk_pipeline = vk::Pipeline::null();
        }
        if self.vk_pipeline_layout != vk::PipelineLayout::null() {
            unsafe {
                self.device.vk_device.destroy_pipeline_layout(
                    self.vk_pipeline_layout,
                    self.device.allocation_callbacks(),
                );
            }
            self.vk_pipeline_layout = vk::PipelineLayout::null();
        }
        self.descriptor_set_layouts.clear();
    }
}

impl Pipeline {
    pub fn builder() -> PipelineBuilder {
        Default::default()
    }

    pub(crate) fn new(device: Arc<Device>, builder: &PipelineBuilder) -> Self {
        let descriptor_set_layouts: Vec<_> =
            builder.descriptor_set_layouts.values().cloned().collect();
        let vk_descriptor_set_layouts: Vec<_> = descriptor_set_layouts
            .iter()
            .map(|set| set.vk_descriptor_set_layout)
            .collect();
        let vk_pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&vk_descriptor_set_layouts)
            .push_constant_ranges(&[])
            .build();
        let vk_pipeline_layout = unsafe {
            device
                .vk_device
                .create_pipeline_layout(
                    &vk_pipeline_layout_create_info,
                    device.allocation_callbacks(),
                )
                .unwrap()
        };

        let shader = builder.shader.clone().unwrap();
        let mut vk_pipeline_shader_stage_create_info_builder =
            shader.get_pipeline_shader_stage_create_info();

        let mut subgroup_size_info = if device.is_dynamic_subgroup_size {
            Some(
                vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::builder()
                    .required_subgroup_size(device.subgroup_size as u32),
            )
        } else {
            None
        };

        if let Some(subgroup_size_info) = &mut subgroup_size_info {
            vk_pipeline_shader_stage_create_info_builder =
                vk_pipeline_shader_stage_create_info_builder.push_next(subgroup_size_info);
        }
        let vk_pipeline_shader_stage_create_info =
            vk_pipeline_shader_stage_create_info_builder.build();

        let vk_compute_pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(vk_pipeline_shader_stage_create_info)
            .layout(vk_pipeline_layout)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1)
            .build();
        let vk_pipelines = unsafe {
            device
                .vk_device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk_compute_pipeline_create_info],
                    device.allocation_callbacks(),
                )
                .unwrap()
        };

        Self {
            device,
            _shader: builder.shader.clone().unwrap(),
            vk_pipeline_layout,
            vk_pipeline: vk_pipelines[0],
            descriptor_set_layouts,
        }
    }
}
