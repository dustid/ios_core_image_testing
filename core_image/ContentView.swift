import SwiftUI
import MetalKit
import AVFoundation
import MetalPerformanceShaders
import CoreMedia

struct ContentView: View {
    @StateObject private var cameraViewModel = CameraViewModel()
    
    var body: some View {
        ZStack {
            CameraPreview(cameraViewModel: cameraViewModel)
            
            if let focusImage = cameraViewModel.focusImage {
                Image(uiImage: focusImage)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: UIScreen.main.bounds.width, height: UIScreen.main.bounds.height)
            }
        }
        .onAppear {
            cameraViewModel.startSession()
        }
        .onDisappear {
            cameraViewModel.stopSession()
        }
    }
}

class CameraViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let device = MTLCreateSystemDefaultDevice()!
    private let commandQueue: MTLCommandQueue
    private let context: CIContext
    private let mpsLaplacian: MPSImageLaplacian
    private var previewLayer: AVCaptureVideoPreviewLayer?
    
    var videoPreviewLayer: AVCaptureVideoPreviewLayer? {
        return previewLayer
    }

    private func setupVideoPreview() {
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer?.connection?.videoOrientation = AVCaptureVideoOrientation.landscapeRight
        // Additional customization for the preview layer if needed
    }
    
    @Published var focusImage: UIImage?
    
    override init() {
        commandQueue = device.makeCommandQueue()!
        context = CIContext(mtlDevice: device)
        mpsLaplacian = MPSImageLaplacian(device: device)
        super.init()
        setupCaptureSession()
        setupVideoPreview()
    }
    
    func startSession() {
        if !captureSession.isRunning {
            captureSession.startRunning()
        }
    }
    
    func stopSession() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }
    
    private func setupCaptureSession() {
        guard let camera = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: camera),
              captureSession.canAddInput(input) else {
            return
        }
        

        
        captureSession.addInput(input)
        
        
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue.global(qos: .userInteractive))
    
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let cgImage = convertCIImageToCGImage(inputImage: ciImage)
        
        let outputImage = laplacian(image: cgImage!)
        
        DispatchQueue.main.async {
            self.focusImage = UIImage(cgImage: outputImage!)
        }
    }
    
    func convertCGImageToCIImage(inputImage: CGImage) -> CIImage! {
        let ciImage = CIImage(cgImage: inputImage)
        return ciImage
    }
    
    func convertCIImageToCGImage(inputImage: CIImage) -> CGImage? {
        let context = CIContext(options: nil)
        if let cgImage = context.createCGImage(inputImage, from: inputImage.extent) {
            return cgImage
        }
        return nil
    }
    
    private func createMetalTexture(from ciImage: CIImage) -> MTLTexture? {
        let bounds = ciImage.extent
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                                         width: Int(bounds.width),
                                                                         height: Int(bounds.height),
                                                                         mipmapped: false)
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let mtlTexture = device.makeTexture(descriptor: textureDescriptor) else {
            return nil
        }
        
        context.render(ciImage, to: mtlTexture, commandBuffer: commandQueue.makeCommandBuffer(), bounds: bounds, colorSpace: CGColorSpaceCreateDeviceRGB())
        
        return mtlTexture
    }

    private func createIntermediateTexture(from sourceTexture: MTLTexture) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: sourceTexture.pixelFormat,
                                                                  width: sourceTexture.width,
                                                                  height: sourceTexture.height,
                                                                  mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let intermediateTexture = device.makeTexture(descriptor: descriptor) else {
            return nil
        }
        
        return intermediateTexture
    }
    
    func laplacian(image: CGImage) -> CGImage? {
        let commandBuffer = self.commandQueue.makeCommandBuffer()!

        let laplacian = MPSImageLaplacian(device: self.device)


        let textureLoader = MTKTextureLoader(device: self.device)
        let options: [MTKTextureLoader.Option : Any]? = nil
        let srcTex = try! textureLoader.newTexture(cgImage: image, options: options)

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: srcTex.pixelFormat,
                                                            width: srcTex.width,
                                                            height: srcTex.height,
                                                            mipmapped: false)
        desc.pixelFormat = .rgba8Unorm
        desc.usage = [.shaderRead, .shaderWrite]

        let lapTex = self.device.makeTexture(descriptor: desc)!

        laplacian.encode(commandBuffer: commandBuffer, sourceTexture: srcTex, destinationTexture: lapTex)

        #if os(macOS)
        let blitCommandEncoder = commandBuffer.makeBlitCommandEncoder()!
        blitCommandEncoder.synchronize(resource: lapTex)
        blitCommandEncoder.endEncoding()
        #endif

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Note: You may want to use a different color space depending
        // on what you're doing with the image
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        // Note: We skip the last component (A) since the Laplacian of the alpha
        // channel of an opaque image is 0 everywhere, and that interacts oddly
        // when we treat the result as an RGBA image.
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue
        let bytesPerRow = lapTex.width * 4
        let bitmapContext = CGContext(data: nil,
                                      width: lapTex.width,
                                      height: lapTex.height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo)!
        

        lapTex.getBytes(bitmapContext.data!,
                        bytesPerRow: bytesPerRow,
                        from: MTLRegionMake2D(0, 0, lapTex.width, lapTex.height),
                        mipmapLevel: 0)
        
        //let im = overlayImageWithTexture(originalImage: image, texture: lapTex)
        
        return bitmapContext.makeImage()
    }
    

    func overlayImageWithTexture(originalImage: CGImage, texture: MTLTexture) -> CGImage? {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return nil
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue
        let bytesPerPixel = 4
        let bytesPerRow = originalImage.width * bytesPerPixel
        let context = CGContext(data: nil,
                                width: originalImage.width,
                                height: originalImage.height,
                                bitsPerComponent: 8,
                                bytesPerRow: bytesPerRow,
                                space: colorSpace,
                                bitmapInfo: bitmapInfo)!
        
        // Draw the original image onto the context
        context.draw(originalImage, in: CGRect(x: 0, y: 0, width: originalImage.width, height: originalImage.height))
        
        // Create a Metal texture descriptor matching the original image properties
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                                         width: originalImage.width,
                                                                         height: originalImage.height,
                                                                         mipmapped: false)
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        // Create a Metal texture from the context's data
        guard let metalTexture = device.makeTexture(descriptor: textureDescriptor) else {
            return nil
        }
        metalTexture.replace(region: MTLRegionMake2D(0, 0, originalImage.width, originalImage.height),
                             mipmapLevel: 0,
                             withBytes: context.data!,
                             bytesPerRow: bytesPerRow)
        
        // Create a Metal library and compute pipeline state
        guard let library = device.makeDefaultLibrary(),
              let kernelFunction = library.makeFunction(name: "overlayBlendKernel"),
              let pipelineState = try? device.makeComputePipelineState(function: kernelFunction) else {
            return nil
        }
        
        // Create a command buffer and encoder
        guard let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        
        // Set up the compute pipeline state
        commandEncoder.setComputePipelineState(pipelineState)
        
        // Set the original image and overlay texture as input textures
        commandEncoder.setTexture(metalTexture, index: 0)
        commandEncoder.setTexture(texture, index: 1)
        
        // Create an output texture for the blended result
        let outputTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                                              width: originalImage.width,
                                                                              height: originalImage.height,
                                                                              mipmapped: false)
        outputTextureDescriptor.usage = [.shaderRead, .shaderWrite]
        let outputTexture = device.makeTexture(descriptor: outputTextureDescriptor)!
        commandEncoder.setTexture(outputTexture, index: 2)
        
        // Set up the thread groups and dispatch the compute kernel
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(width: (originalImage.width + threadGroupSize.width - 1) / threadGroupSize.width,
                                   height: (originalImage.height + threadGroupSize.height - 1) / threadGroupSize.height,
                                   depth: 1)
        commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        
        // End encoding the command encoder and commit the command buffer
        commandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Retrieve the modified Metal texture data
        let modifiedBytesPerRow = outputTexture.width * bytesPerPixel
        let modifiedData = UnsafeMutablePointer<UInt8>.allocate(capacity: outputTexture.height * modifiedBytesPerRow)
        outputTexture.getBytes(modifiedData,
                               bytesPerRow: modifiedBytesPerRow,
                               from: MTLRegionMake2D(0, 0, outputTexture.width, outputTexture.height),
                               mipmapLevel: 0)
        
        // Copy the modified texture data to the context
        context.data?.copyMemory(from: modifiedData, byteCount: outputTexture.height * modifiedBytesPerRow)
        
        // Create a CGImage from the modified context
        let modifiedImage = context.makeImage()
        
        // Free the allocated memory
        modifiedData.deallocate()
        
        return modifiedImage
    }



}

struct CameraPreview: UIViewRepresentable {
    @ObservedObject var cameraViewModel: CameraViewModel
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let layer = cameraViewModel.videoPreviewLayer
        layer?.frame = view.layer.bounds
        view.layer.addSublayer(layer!)
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

