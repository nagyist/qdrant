use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use gpu_allocator::MemoryLocation;

use crate::*;

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum BufferType {
    Uniform,
    Storage,
    CpuToGpu,
    GpuToCpu,
}

pub struct Buffer {
    pub device: Arc<Device>,
    pub vk_buffer: vk::Buffer,
    pub buffer_type: BufferType,
    pub size: usize,
    pub allocation: Mutex<Allocation>,
}

impl Resource for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        let mut allocation = Mutex::new(Allocation::default());
        std::mem::swap(&mut allocation, &mut self.allocation);
        let allocation = allocation.into_inner().unwrap();
        self.device.gpu_free(allocation);
        unsafe {
            self.device
                .vk_device
                .destroy_buffer(self.vk_buffer, self.device.allocation_callbacks())
        };
        self.size = 0;
        self.vk_buffer = vk::Buffer::null();
    }
}

impl Buffer {
    pub fn new(device: Arc<Device>, buffer_type: BufferType, size: usize) -> GpuResult<Self> {
        let (usage_flags, location) = match buffer_type {
            BufferType::Uniform => (
                vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
            ),
            BufferType::Storage => (
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
            ),
            BufferType::CpuToGpu => (vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu),
            BufferType::GpuToCpu => (vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu),
        };
        let vk_info = vk::BufferCreateInfo::builder()
            .size(size as vk::DeviceSize)
            .usage(usage_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let vk_buffer;
        let allocation;
        {
            vk_buffer = unsafe { device.vk_device.create_buffer(&vk_info, None) }.unwrap();
            let requirements =
                unsafe { device.vk_device.get_buffer_memory_requirements(vk_buffer) };

            let allocation_result = device.gpu_alloc(&AllocationCreateDesc {
                name: "",
                requirements,
                location,
                linear: true, // Buffers are always linear
            });

            allocation = match allocation_result {
                Ok(allocation) => allocation,
                Err(e) => {
                    unsafe {
                        device
                            .vk_device
                            .destroy_buffer(vk_buffer, device.allocation_callbacks());
                    }
                    return Err(e);
                }
            };

            unsafe {
                device
                    .vk_device
                    .bind_buffer_memory(vk_buffer, allocation.memory(), allocation.offset())
                    .unwrap()
            };
        }

        Ok(Self {
            device,
            vk_buffer,
            buffer_type,
            size,
            allocation: Mutex::new(allocation),
        })
    }

    pub fn download<T: Sized>(&self, data: &mut T, offset: usize) {
        if self.buffer_type != BufferType::GpuToCpu {
            panic!("Download works only for buffers with GpuToCpu type");
        }
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(
                (data as *mut T) as *mut u8,
                std::mem::size_of::<T>(),
            );
            self.download_bytes(bytes, offset);
        }
    }

    pub fn download_slice<T: Sized>(&self, data: &mut [T], offset: usize) {
        if self.buffer_type != BufferType::GpuToCpu {
            panic!("Download works only for buffers with GpuToCpu type");
        }
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(
                (data.as_ptr() as *mut T) as *mut u8,
                std::mem::size_of_val(data),
            );
            self.download_bytes(bytes, offset);
        }
    }

    pub fn download_bytes(&self, data: &mut [u8], offset: usize) {
        if self.buffer_type != BufferType::GpuToCpu {
            panic!("Download works only for buffers with GpuToCpu type");
        }
        unsafe {
            let allocation = self.allocation.lock().unwrap();
            let slice = allocation.mapped_slice().unwrap();
            let ptr = slice.as_ptr().add(offset);
            // TODD(gpu): check ranges
            std::ptr::copy(ptr, data.as_mut_ptr(), data.len());
        }
    }

    pub fn upload<T: Sized>(&self, data: &T, offset: usize) {
        unsafe {
            let bytes = std::slice::from_raw_parts(
                (data as *const T) as *const u8,
                std::mem::size_of::<T>(),
            );
            self.upload_bytes(bytes, offset);
        }
    }

    pub fn upload_slice<T: Sized>(&self, data: &[T], offset: usize) {
        unsafe {
            let mut allocation = self.allocation.lock().unwrap();
            let slice = allocation.mapped_slice_mut().unwrap();
            let ptr = slice.as_mut_ptr().add(offset);
            // TODD(gpu): check ranges
            std::ptr::copy(data.as_ptr() as *const u8, ptr, std::mem::size_of_val(data));
        }
    }

    pub fn upload_bytes(&self, data: &[u8], offset: usize) {
        unsafe {
            let mut allocation = self.allocation.lock().unwrap();
            let slice = allocation.mapped_slice_mut().unwrap();
            let ptr = slice.as_mut_ptr().add(offset);
            // TODD(gpu): check ranges
            std::ptr::copy(data.as_ptr(), ptr, data.len());
        }
    }
}
