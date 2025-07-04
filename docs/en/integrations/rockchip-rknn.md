---
comments: true
description: Learn how to export YOLO11 models to RKNN format for efficient deployment on Rockchip platforms with enhanced performance.
keywords: YOLO11, RKNN, model export, Ultralytics, Rockchip, machine learning, model deployment, computer vision, deep learning, edge AI, NPU, embedded devices
---

# Rockchip RKNN Export for Ultralytics YOLO11 Models

When deploying computer vision models on embedded devices, especially those powered by Rockchip processors, having a compatible model format is essential. Exporting [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models to RKNN format ensures optimized performance and compatibility with Rockchip's hardware. This guide will walk you through converting your YOLO11 models to RKNN format, enabling efficient deployment on Rockchip platforms.

<p align="center">
  <img width="50%" src="https://github.com/ultralytics/assets/releases/download/v0.0.0/rockchip-rknn.avif" alt="RKNN">
</p>

!!! note

    This guide has been tested with [Radxa Rock 5B](https://radxa.com/products/rock5/5b/) which is based on Rockchip RK3588 and [Radxa Zero 3W](https://radxa.com/products/zeros/zero3w/) which is based on Rockchip RK3566. It is expected to work across other Rockchip-based devices which supports [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) such as RK3576, RK3568, RK3562, RV1103, RV1106, RV1103B, RV1106B and RK2118.

## What is Rockchip?

Renowned for delivering versatile and power-efficient solutions, Rockchip designs advanced System-on-Chips (SoCs) that power a wide range of consumer electronics, industrial applications, and AI technologies. With ARM-based architecture, built-in Neural Processing Units (NPUs), and high-resolution multimedia support, Rockchip SoCs enable cutting-edge performance for devices like tablets, smart TVs, IoT systems, and [edge AI applications](https://www.ultralytics.com/blog/understanding-the-real-world-applications-of-edge-ai). Companies like Radxa, ASUS, Pine64, Orange Pi, Odroid, Khadas, and Banana Pi offer a variety of products based on Rockchip SoCs, further extending their reach and impact across diverse markets.

## RKNN Toolkit

The [RKNN Toolkit](https://github.com/airockchip/rknn-toolkit2) is a set of tools and libraries provided by Rockchip to facilitate the deployment of deep learning models on their hardware platforms. RKNN, or Rockchip Neural Network, is the proprietary format used by these tools. RKNN models are designed to take full advantage of the hardware acceleration provided by Rockchip's NPU (Neural Processing Unit), ensuring high performance in AI tasks on devices like RK3588, RK3566, RV1103, RV1106, and other Rockchip-powered systems.

## Key Features of RKNN Models

RKNN models offer several advantages for deployment on Rockchip platforms:

- **Optimized for NPU**: RKNN models are specifically optimized to run on Rockchip's NPUs, ensuring maximum performance and efficiency.
- **Low Latency**: The RKNN format minimizes inference latency, which is critical for real-time applications on edge devices.
- **Platform-Specific Customization**: RKNN models can be tailored to specific Rockchip platforms, enabling better utilization of hardware resources.
- **Power Efficiency**: By leveraging dedicated NPU hardware, RKNN models consume less power than CPU or GPU-based processing, extending battery life for portable devices.

## Flash OS to Rockchip hardware

The first step after getting your hands on a Rockchip-based device is to flash an OS so that the hardware can boot into a working environment. In this guide we will point to getting started guides of the two devices that we tested which are Radxa Rock 5B and Radxa Zero 3W.

- [Radxa Rock 5B Getting Started Guide](https://docs.radxa.com/en/rock5/rock5b)
- [Radxa Zero 3W Getting Started Guide](https://docs.radxa.com/en/zero/zero3)

## Export to RKNN: Converting Your YOLO11 Model

Export an Ultralytics YOLO11 model to RKNN format and run inference with the exported model.

!!! note

    Make sure to use an X86-based Linux PC to export the model to RKNN because exporting on Rockchip-based devices (ARM64) are not supported.

### Installation

To install the required packages, run:

!!! Tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO11
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

!!! note

    Export is currently only supported for detection models. More model support will be coming in the future.

!!! Example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to RKNN format
        # 'name' can be one of rk3588, rk3576, rk3566, rk3568, rk3562, rv1103, rv1106, rv1103b, rv1106b, rk2118
        model.export(format="rknn", name="rk3588")  # creates '/yolo11n_rknn_model'
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to RKNN format
        # 'name' can be one of rk3588, rk3576, rk3566, rk3568, rk3562, rv1103, rv1106, rv1103b, rv1106b, rk2118
        yolo export model=yolo11n.pt format=rknn name=rk3588 # creates '/yolo11n_rknn_model'
        ```

### Export Arguments

| Argument | Type             | Default    | Description                                                                                                                             |
| -------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'rknn'`   | Target format for the exported model, defining compatibility with various deployment environments.                                      |
| `imgsz`  | `int` or `tuple` | `640`      | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.       |
| `batch`  | `int`            | `1`        | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `name`   | `str`            | `'rk3588'` | Specifies the Rockchip model (rk3588, rk3576, rk3566, rk3568, rk3562, rv1103, rv1106, rv1103b, rv1106b, rk2118)                         |
| `device` | `str`            | `None`     | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`).                                                               |

!!! tip

    Please make sure to use an x86 Linux machine when exporting to RKNN.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO11 RKNN Models

Once you've successfully exported your Ultralytics YOLO11 models to RKNN format, the next step is deploying these models on Rockchip-based devices.

### Installation

To install the required packages, run:

!!! Tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO11
        pip install ultralytics
        ```

### Usage

!!! Example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported RKNN model
        rknn_model = YOLO("./yolo11n_rknn_model")

        # Run inference
        results = rknn_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported model
        yolo predict model='./yolo11n_rknn_model' source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    If you encounter a log message indicating that the RKNN runtime version does not match the RKNN Toolkit version and the inference fails, please replace `/usr/lib/librknnrt.so` with official [librknnrt.so file](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so).

    ![RKNN export screenshot](https://github.com/ultralytics/assets/releases/download/v0.0.0/rknn-npu-log.avif)

## Real-World Applications

Rockchip-powered devices with YOLO11 RKNN models can be used in various applications:

- **Smart Surveillance**: Deploy efficient object detection systems for security monitoring with low power consumption.
- **Industrial Automation**: Implement quality control and defect detection directly on embedded devices.
- **Retail Analytics**: Track customer behavior and inventory management in real-time without cloud dependency.
- **Smart Agriculture**: Monitor crop health and detect pests using [computer vision in agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture).
- **Autonomous Robotics**: Enable vision-based navigation and obstacle detection on resource-constrained platforms.

## Benchmarks

YOLO11 benchmarks below were run by the Ultralytics team on Radxa Rock 5B based on Rockchip RK3588 with `rknn` model format measuring speed and accuracy.

!!! tip "Performance"

    | Model   | Format | Status | Size (MB) | mAP50-95(B) | Inference time (ms/im) |
    | ------- | ------ | ------ | --------- | ----------- | ---------------------- |
    | YOLO11n | `rknn` | ✅     | 7.4       | 0.505       | 71.5                   |
    | YOLO11s | `rknn` | ✅     | 20.7      | 0.578       | 98.9                   |
    | YOLO11m | `rknn` | ✅     | 41.9      | 0.629       | 235.3                  |
    | YOLO11l | `rknn` | ✅     | 53.3      | 0.633       | 282.0                  |
    | YOLO11x | `rknn` | ✅     | 114.6     | 0.687       | 679.2                  |

    Benchmarked with `ultralytics 8.3.152`

    !!! note

        Validation for the above benchmarks were done using COCO18 dataset. Inference time does not include pre/ post-processing.

## Summary

In this guide, you've learned how to export Ultralytics YOLO11 models to RKNN format to enhance their deployment on Rockchip platforms. You were also introduced to the RKNN Toolkit and the specific advantages of using RKNN models for edge AI applications.

The combination of [Ultralytics YOLO11](https://www.ultralytics.com/blog/all-you-need-to-know-about-ultralytics-yolo11-and-its-applications) and Rockchip's NPU technology provides an efficient solution for running advanced computer vision tasks on embedded devices. This approach enables real-time [object detection](https://www.ultralytics.com/blog/a-guide-to-deep-dive-into-object-detection-in-2025) and other vision AI applications with minimal power consumption and high performance.

For further details on usage, visit the [RKNN official documentation](https://github.com/airockchip/rknn-toolkit2).

Also, if you'd like to know more about other Ultralytics YOLO11 integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of useful resources and insights there.

## FAQ

### How do I export my Ultralytics YOLO model to RKNN format?

You can easily export your Ultralytics YOLO model to RKNN format using the `export()` method in the Ultralytics Python package or via the command-line interface (CLI). Ensure you are using an x86-based Linux PC for the export process, as ARM64 devices like Rockchip are not supported for this operation. You can specify the target Rockchip platform using the `name` argument, such as `rk3588`, `rk3566`, or others. This process generates an optimized RKNN model ready for deployment on your Rockchip device, taking advantage of its Neural Processing Unit (NPU) for accelerated inference.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load your YOLO model
        model = YOLO("yolo11n.pt")

        # Export to RKNN format for a specific Rockchip platform
        model.export(format="rknn", name="rk3588")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=rknn name=rk3588
        ```

### What are the benefits of using RKNN models on Rockchip devices?

RKNN models are specifically designed to leverage the hardware acceleration capabilities of Rockchip's Neural Processing Units (NPUs). This optimization results in significantly faster inference speeds and reduced latency compared to running generic model formats like ONNX or TensorFlow Lite on the same hardware. Using RKNN models allows for more efficient use of the device's resources, leading to lower power consumption and better overall performance, especially critical for real-time applications on edge devices. By converting your Ultralytics YOLO models to RKNN, you can achieve optimal performance on devices powered by Rockchip SoCs like the RK3588, RK3566, and others.

### Can I deploy RKNN models on devices from other manufacturers like NVIDIA or Google?

RKNN models are specifically optimized for Rockchip platforms and their integrated NPUs. While you can technically run an RKNN model on other platforms using software emulation, you will not benefit from the hardware acceleration provided by Rockchip devices. For optimal performance on other platforms, it's recommended to export your Ultralytics YOLO models to formats specifically designed for those platforms, such as TensorRT for NVIDIA GPUs or [TensorFlow Lite](https://docs.ultralytics.com/integrations/tflite/) for Google's Edge TPU. Ultralytics supports exporting to a wide range of formats, ensuring compatibility with various hardware accelerators.

### What Rockchip platforms are supported for RKNN model deployment?

The Ultralytics YOLO export to RKNN format supports a wide range of Rockchip platforms, including the popular RK3588, RK3576, RK3566, RK3568, RK3562, RV1103, RV1106, RV1103B, RV1106B, and RK2118. These platforms are commonly found in devices from manufacturers like Radxa, ASUS, Pine64, Orange Pi, Odroid, Khadas, and Banana Pi. This broad support ensures that you can deploy your optimized RKNN models on various Rockchip-powered devices, from single-board computers to industrial systems, taking full advantage of their AI acceleration capabilities for enhanced performance in your computer vision applications.

### How does the performance of RKNN models compare to other formats on Rockchip devices?

RKNN models generally outperform other formats like ONNX or TensorFlow Lite on Rockchip devices due to their optimization for Rockchip's NPUs. For instance, benchmarks on the Radxa Rock 5B (RK3588) show that [YOLO11n](https://www.ultralytics.com/blog/all-you-need-to-know-about-ultralytics-yolo11-and-its-applications) in RKNN format achieves an inference time of 99.5 ms/image, significantly faster than other formats. This performance advantage is consistent across various YOLO11 model sizes, as demonstrated in the [benchmarks section](#benchmarks). By leveraging the dedicated NPU hardware, RKNN models minimize latency and maximize throughput, making them ideal for real-time applications on Rockchip-based edge devices.
