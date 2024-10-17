use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use gpu_allocator::MemoryLocation;

use crate::*;

static DOWNLOAD_NOT_ALLOWED_ERROR: &str = "Download works only for buffers with GpuToCpu type";
static UPLOAD_NOT_ALLOWED_ERROR: &str = "Upload works only for buffers with CpuToGpu type";

/// Buffer is a GPU resource that represents a linear memory region.
pub struct Buffer {
    /// Device that owns the buffer.
    pub device: Arc<Device>,

    /// Vulkan buffer handle.
    pub vk_buffer: vk::Buffer,

    /// Buffer type. It defines how the buffer can be used.
    pub buffer_type: BufferType,

    /// Buffer size in bytes.
    pub size: usize,

    /// GPU memory allocation that backs the buffer.
    pub allocation: Mutex<Allocation>,
}

/// Buffer type defines how the buffer can be used.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum BufferType {
    /// Uniform data for a shader.
    Uniform,

    /// Storage buffer can be used as a large read/write buffer.
    Storage,

    /// CpuToGpu buffer can be used as a source for transfer operations only.
    CpuToGpu,

    /// GpuToCpu buffer can be used as a destination for transfer operations only.
    GpuToCpu,
}

// Mark `Buffer` as a GPU resource that should be kept alive while it's in use by the GPU context.
impl Resource for Buffer {}

impl Buffer {
    pub fn new(
        device: Arc<Device>,   // Device that owns the buffer.
        name: impl AsRef<str>, // Name of the buffer for tracking and debugging purposes.
        buffer_type: BufferType,
        size: usize,
    ) -> GpuResult<Arc<Self>> {
        // Vulkan API requires buffer usage flags to be specified during the buffer creation.
        let vk_usage_flags = match buffer_type {
            BufferType::Uniform => {
                vk::BufferUsageFlags::UNIFORM_BUFFER // mark as uniform buffer.
                    | vk::BufferUsageFlags::TRANSFER_DST // For uploading.
                    | vk::BufferUsageFlags::TRANSFER_SRC // For downloading.
            }
            BufferType::Storage => {
                vk::BufferUsageFlags::STORAGE_BUFFER // mark as storage buffer.
                    | vk::BufferUsageFlags::TRANSFER_DST // For uploading.
                    | vk::BufferUsageFlags::TRANSFER_SRC // For downloading.
            }
            // CpuToGpu buffer can be used as a source for transfer operations only.
            BufferType::CpuToGpu => vk::BufferUsageFlags::TRANSFER_SRC,
            // GpuToCpu buffer can be used as a destination for transfer operations only.
            BufferType::GpuToCpu => vk::BufferUsageFlags::TRANSFER_DST,
        };

        // Memory location depends on the buffer type.
        let location = match buffer_type {
            // Allocate Uniform/Storage buffers in GPU memory only.
            BufferType::Uniform => MemoryLocation::GpuOnly,
            BufferType::Storage => MemoryLocation::GpuOnly,
            // Transfer buffers will be visible to both CPU and GPU.
            BufferType::CpuToGpu => MemoryLocation::CpuToGpu,
            BufferType::GpuToCpu => MemoryLocation::GpuToCpu,
        };

        // Create a Vulkan buffer.
        let vk_info = vk::BufferCreateInfo::builder()
            .size(size as vk::DeviceSize)
            .usage(vk_usage_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let vk_buffer = unsafe {
            device
                .vk_device
                .create_buffer(&vk_info, device.allocation_callbacks())
                .unwrap()
        };

        // Allocate memory for the buffer.
        let buffer_allocation_requirements =
            unsafe { device.vk_device.get_buffer_memory_requirements(vk_buffer) };
        let allocation_result = device.gpu_alloc(&AllocationCreateDesc {
            name: name.as_ref(),
            requirements: buffer_allocation_requirements,
            location,
            linear: true, // Buffers are always linear.
        });

        // Check if the allocation was successful.
        let allocation = match allocation_result {
            Ok(allocation) => allocation,
            Err(e) => {
                unsafe {
                    // Because vulkan buffers lifetime is managed manually,
                    // we need to destroy the buffer in case of an allocation error.
                    device
                        .vk_device
                        .destroy_buffer(vk_buffer, device.allocation_callbacks());
                }
                return Err(e);
            }
        };

        // Bind the buffer to the allocated memory.
        unsafe {
            device
                .vk_device
                .bind_buffer_memory(vk_buffer, allocation.memory(), allocation.offset())
                .unwrap()
        };

        Ok(Arc::new(Self {
            device,
            vk_buffer,
            buffer_type,
            size,
            allocation: Mutex::new(allocation),
        }))
    }

    /// Download data from the buffer to the RAM.
    pub fn download<T: Sized>(&self, data: &mut T, offset: usize) -> GpuResult<()> {
        if self.buffer_type != BufferType::GpuToCpu {
            return Err(GpuError::Other(DOWNLOAD_NOT_ALLOWED_ERROR.to_string()));
        }

        unsafe {
            let bytes = std::slice::from_raw_parts_mut(
                (data as *mut T) as *mut u8,
                std::mem::size_of::<T>(),
            );
            self.download_bytes(bytes, offset)
        }
    }

    /// Download data from the buffer to the RAM.
    pub fn download_slice<T: Sized>(&self, data: &mut [T], offset: usize) -> GpuResult<()> {
        if self.buffer_type != BufferType::GpuToCpu {
            return Err(GpuError::Other(DOWNLOAD_NOT_ALLOWED_ERROR.to_string()));
        }

        unsafe {
            let bytes = std::slice::from_raw_parts_mut(
                (data.as_ptr() as *mut T) as *mut u8,
                std::mem::size_of_val(data),
            );
            self.download_bytes(bytes, offset)
        }
    }

    /// Download data from the buffer to the RAM.
    pub fn download_bytes(&self, data: &mut [u8], offset: usize) -> GpuResult<()> {
        if self.buffer_type != BufferType::GpuToCpu {
            return Err(GpuError::Other(DOWNLOAD_NOT_ALLOWED_ERROR.to_string()));
        }

        unsafe {
            let allocation = self.allocation.lock().unwrap();
            let slice = allocation.mapped_slice().unwrap();
            let ptr = slice.as_ptr().add(offset);
            // TODD(gpu): check ranges
            std::ptr::copy(ptr, data.as_mut_ptr(), data.len());
            Ok(())
        }
    }

    /// Upload data from the RAM to the buffer.
    pub fn upload<T: Sized>(&self, data: &T, offset: usize) -> GpuResult<()> {
        if self.buffer_type != BufferType::CpuToGpu {
            return Err(GpuError::Other(UPLOAD_NOT_ALLOWED_ERROR.to_string()));
        }

        unsafe {
            let bytes = std::slice::from_raw_parts(
                (data as *const T) as *const u8,
                std::mem::size_of::<T>(),
            );
            self.upload_bytes(bytes, offset)
        }
    }

    /// Upload data from the RAM to the buffer.
    pub fn upload_slice<T: Sized>(&self, data: &[T], offset: usize) -> GpuResult<()> {
        if self.buffer_type != BufferType::CpuToGpu {
            return Err(GpuError::Other(UPLOAD_NOT_ALLOWED_ERROR.to_string()));
        }

        unsafe {
            let mut allocation = self.allocation.lock().unwrap();
            let slice = allocation.mapped_slice_mut().unwrap();
            let ptr = slice.as_mut_ptr().add(offset);
            // TODD(gpu): check ranges
            std::ptr::copy(data.as_ptr() as *const u8, ptr, std::mem::size_of_val(data));
            Ok(())
        }
    }

    /// Upload data from the RAM to the buffer.
    pub fn upload_bytes(&self, data: &[u8], offset: usize) -> GpuResult<()> {
        if self.buffer_type != BufferType::CpuToGpu {
            return Err(GpuError::Other(UPLOAD_NOT_ALLOWED_ERROR.to_string()));
        }

        unsafe {
            let mut allocation = self.allocation.lock().unwrap();
            let slice = allocation.mapped_slice_mut().unwrap();
            let ptr = slice.as_mut_ptr().add(offset);
            // TODD(gpu): check ranges
            std::ptr::copy(data.as_ptr(), ptr, data.len());
            Ok(())
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // Drop the allocation and free the allocated memory
        let mut allocation = Mutex::new(Allocation::default());
        std::mem::swap(&mut allocation, &mut self.allocation);
        let allocation = allocation.into_inner().unwrap();
        self.device.gpu_free(allocation);

        // Destroy the buffer
        unsafe {
            self.device
                .vk_device
                .destroy_buffer(self.vk_buffer, self.device.allocation_callbacks())
        };

        // Reset the buffer state
        self.size = 0;
        self.vk_buffer = vk::Buffer::null();
    }
}
