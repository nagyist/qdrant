pub mod allocation_callbacks;
pub use allocation_callbacks::*;

pub mod context;
pub use context::*;

pub mod debug_messenger;
pub use debug_messenger::*;

pub mod descriptor_set;
pub use descriptor_set::*;

pub mod descriptor_set_layout;
pub use descriptor_set_layout::*;

pub mod buffer;
pub use buffer::*;

pub mod device;
pub use device::*;

pub mod instance;
pub use instance::*;

pub mod pipeline;
pub use pipeline::*;

pub mod pipeline_builder;
pub use pipeline_builder::*;

pub mod shader;
pub use shader::*;

pub trait Resource: Send + Sync {}

#[derive(Debug)]
pub enum GpuError {
    AllocationError(gpu_allocator::AllocationError),
    NotSupported,
}

pub type GpuResult<T> = Result<T, GpuError>;
