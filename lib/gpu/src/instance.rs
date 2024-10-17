use std::ffi::{c_void, CString};
use std::ptr;

use ash::extensions::ext::DebugUtils;
use ash::vk;

use crate::*;

pub struct Instance {
    _entry: ash::Entry,
    pub vk_instance: ash::Instance,
    pub vk_physical_devices: Vec<PhysicalDevice>,
    pub allocation_callbacks: Option<Box<dyn AllocationCallbacks>>,
    pub layers: Vec<String>,
    pub extensions: Vec<String>,
    vk_debug_utils_loader: Option<ash::extensions::ext::DebugUtils>,
    vk_debug_messenger: vk::DebugUtilsMessengerEXT,
}

#[derive(Clone)]
pub struct PhysicalDevice {
    pub vk_physical_device: vk::PhysicalDevice,
    pub name: String,
}

impl Instance {
    pub fn new(
        name: &str,
        debug_messenger: Option<&dyn DebugMessenger>,
        allocation_callbacks: Option<Box<dyn AllocationCallbacks>>,
        dump_api: bool,
    ) -> GpuResult<Self> {
        unsafe {
            let entry = ash::Entry::load().unwrap();
            let app_name = CString::new(name).unwrap();
            let engine_name = CString::new(name).unwrap();
            let app_info = vk::ApplicationInfo {
                s_type: vk::StructureType::APPLICATION_INFO,
                p_next: ptr::null(),
                p_application_name: app_name.as_ptr(),
                application_version: 0,
                p_engine_name: engine_name.as_ptr(),
                engine_version: 0,
                api_version: vk::make_api_version(0, 1, 3, 0),
            };

            let extensions = Self::get_extensions_list(debug_messenger.is_some());
            let extensions_cstr: Vec<CString> = extensions
                .iter()
                .map(|s| CString::new(s.clone().into_bytes()).unwrap())
                .collect();
            let extension_names_raw: Vec<*const i8> = extensions_cstr
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let layers = Self::get_layers_list(debug_messenger.is_some(), dump_api);
            let layers_cstr: Vec<CString> = layers
                .iter()
                .map(|s| CString::new(s.clone().into_bytes()).unwrap())
                .collect();
            let layers_raw: Vec<*const i8> = layers_cstr
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let debug_utils_create_info = debug_messenger.map(Self::debug_messenger_create_info);
            let create_info_p_next = if let Some(debug_utils_create_info) = &debug_utils_create_info
            {
                debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                    as *const c_void
            } else {
                ptr::null()
            };

            let create_flags = if cfg!(any(target_os = "macos")) {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::default()
            };

            let create_info = vk::InstanceCreateInfo {
                s_type: vk::StructureType::INSTANCE_CREATE_INFO,
                p_next: create_info_p_next,
                flags: create_flags,
                p_application_info: &app_info,
                pp_enabled_layer_names: layers_raw.as_ptr(),
                enabled_layer_count: layers_raw.len() as u32,
                pp_enabled_extension_names: extension_names_raw.as_ptr(),
                enabled_extension_count: extension_names_raw.len() as u32,
            };

            let vk_allocation_callbacks = allocation_callbacks
                .as_ref()
                .map(|a| a.allocation_callbacks());
            let vk_instance: ash::Instance = entry
                .create_instance(&create_info, vk_allocation_callbacks)
                .expect("Failed to create instance!");

            let (vk_debug_utils_loader, vk_debug_messenger) = if let Some(debug_messenger) =
                debug_messenger
            {
                let debug_utils_loader =
                    ash::extensions::ext::DebugUtils::new(&entry, &vk_instance);
                let messenger_create_info = Self::debug_messenger_create_info(debug_messenger);
                let utils_messenger = debug_utils_loader
                    .create_debug_utils_messenger(&messenger_create_info, vk_allocation_callbacks)
                    .expect("Debug Utils Callback");
                (Some(debug_utils_loader), utils_messenger)
            } else {
                (None, vk::DebugUtilsMessengerEXT::null())
            };

            let vk_physical_devices = vk_instance.enumerate_physical_devices().unwrap();
            let vk_physical_devices = vk_physical_devices
                .iter()
                .map(|vk_physical_device| {
                    let device_properties =
                        vk_instance.get_physical_device_properties(*vk_physical_device);
                    log::debug!(
                        "Foung GPU device: {:?}",
                        ::std::ffi::CStr::from_ptr(device_properties.device_name.as_ptr())
                    );
                    let props = vk_instance.get_physical_device_properties(*vk_physical_device);
                    let device_name = ::std::ffi::CStr::from_ptr(props.device_name.as_ptr());
                    let device_name = device_name.to_str().unwrap().to_owned();
                    log::info!("Foung GPU device: {device_name}");
                    PhysicalDevice {
                        vk_physical_device: *vk_physical_device,
                        name: device_name,
                    }
                })
                .collect::<Vec<_>>();

            if vk_physical_devices.is_empty() {
                panic!("No physical device found");
            }

            Ok(Self {
                _entry: entry,
                vk_instance,
                vk_physical_devices,
                allocation_callbacks,
                layers,
                extensions,
                vk_debug_utils_loader,
                vk_debug_messenger,
            })
        }
    }

    fn debug_messenger_create_info(
        debug_messenger: &dyn DebugMessenger,
    ) -> vk::DebugUtilsMessengerCreateInfoEXT {
        vk::DebugUtilsMessengerCreateInfoEXT {
            s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            p_next: ptr::null(),
            flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
            message_severity: debug_messenger.get_severity_flags(),
            message_type: debug_messenger.get_message_type_flags(),
            pfn_user_callback: debug_messenger.get_callback(),
            p_user_data: ptr::null_mut(),
        }
    }

    pub fn is_validation_enable(&self) -> bool {
        self.vk_debug_utils_loader.is_some()
    }

    pub fn allocation_callbacks(&self) -> Option<&vk::AllocationCallbacks> {
        self.allocation_callbacks
            .as_ref()
            .map(|alloc| alloc.allocation_callbacks())
    }

    fn get_layers_list(validation: bool, dump_api: bool) -> Vec<String> {
        let mut result = Vec::new();
        if validation {
            result.push("VK_LAYER_KHRONOS_validation".to_owned());
        }
        if dump_api {
            result.push("VK_LAYER_LUNARG_api_dump".to_owned());
        }
        result
    }

    fn get_extensions_list(validation: bool) -> Vec<String> {
        let mut extensions_list = Vec::new();
        if validation {
            extensions_list.push(DebugUtils::name().to_str().unwrap().to_string());
        }

        #[cfg(target_os = "macos")]
        {
            extensions_list.push(
                vk::KhrPortabilityEnumerationFn::name()
                    .to_str()
                    .unwrap()
                    .to_string(),
            );
            extensions_list.push(
                vk::KhrGetPhysicalDeviceProperties2Fn::name()
                    .to_str()
                    .unwrap()
                    .to_string(),
            );
        }
        extensions_list
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        let allocation_callbacks = self.allocation_callbacks();
        unsafe {
            if let Some(loader) = &self.vk_debug_utils_loader {
                if self.vk_debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                    loader.destroy_debug_utils_messenger(
                        self.vk_debug_messenger,
                        allocation_callbacks,
                    );
                }
            }
            self.vk_instance.destroy_instance(allocation_callbacks);
        }
    }
}
