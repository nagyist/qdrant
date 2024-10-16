use ash::vk;

pub trait AllocationCallbacks: Send + Sync + 'static {
    fn allocation_callbacks(&self) -> &vk::AllocationCallbacks;
}
