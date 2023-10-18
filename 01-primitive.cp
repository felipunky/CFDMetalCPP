/*
 *
 * Copyright 2022 Apple Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <iostream>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <AppKit/AppKit.hpp>
#include <MetalKit/MetalKit.hpp>
#include <chrono>

#include <simd/simd.h>

#include <ApplicationServices/ApplicationServices.h>

#pragma region Declarations {

const uint32_t maxFramesInFlight = 3;
const uint32_t width = 512,
               height = 512;

struct UBO
{
    simd::float2 iResolution;
    simd::float2 iMouse;
    float iTime;
    float iTimeDelta;
};

struct cUBO
{
    simd::float2 iResolution;
    simd::float2 iMouse;
    simd::float2 iVelocity;
    float iTime;
    float iTimeDelta;
};

class Renderer
{
    public:
        Renderer( MTL::Device* pDevice );
        ~Renderer();
        void buildShaders();
        void buildComputePipeline();
        void buildTextures();
        void buildBuffers();
        void generateComputeTexture( MTL::CommandBuffer* pCommandBuffer );
        void draw( MTK::View* pView );

    private:
        MTL::Device* _pDevice;
        MTL::CommandQueue* _pCommandQueue;
        MTL::RenderPipelineState* _pPSO;
        MTL::ComputePipelineState* _cPSO;
        MTL::Texture* _pTexture;
        MTL::Texture* _pBackBuffer;
        MTL::Buffer* _pVertexPositionsBuffer;
        MTL::Buffer* _pVertexColorsBuffer;
        MTL::Buffer* _pIndexBuffer;
        MTL::Buffer* _pUniformBuffer[maxFramesInFlight];
        MTL::Buffer* _pComputeUBOBuffer;
        dispatch_semaphore_t _semaphore;
        uint32_t _frame;
        simd::float2 _iMouse;
        simd::float2 _iMouseLast;
        simd::float2 _iVelocity;
        float _iTime = 0.0f;
        float _iTimeDelta = 0.0f;
        float _iLastTime = 0.0f;
};

class MyMTKViewDelegate : public MTK::ViewDelegate
{
    public:
        MyMTKViewDelegate( MTL::Device* pDevice );
        virtual ~MyMTKViewDelegate() override;
        virtual void drawInMTKView( MTK::View* pView ) override;

    private:
        Renderer* _pRenderer;
};

class MyAppDelegate : public NS::ApplicationDelegate
{
    public:
        ~MyAppDelegate();

        NS::Menu* createMenuBar();

        virtual void applicationWillFinishLaunching( NS::Notification* pNotification ) override;
        virtual void applicationDidFinishLaunching( NS::Notification* pNotification ) override;
        virtual bool applicationShouldTerminateAfterLastWindowClosed( NS::Application* pSender ) override;

    private:
        NS::Window* _pWindow;
        MTK::View* _pMtkView;
        MTL::Device* _pDevice;
        MyMTKViewDelegate* _pViewDelegate = nullptr;
};

#pragma endregion Declarations }


int main( int argc, char* argv[] )
{
    NS::AutoreleasePool* pAutoreleasePool = NS::AutoreleasePool::alloc()->init();

    MyAppDelegate del;
    
    NS::Application* pSharedApplication = NS::Application::sharedApplication();
    pSharedApplication->setDelegate( &del );
    pSharedApplication->run();

    pAutoreleasePool->release();

    return 0;
}


#pragma mark - AppDelegate
#pragma region AppDelegate {

MyAppDelegate::~MyAppDelegate()
{
    _pMtkView->release();
    _pWindow->release();
    _pDevice->release();
    delete _pViewDelegate;
}

NS::Menu* MyAppDelegate::createMenuBar()
{
    using NS::StringEncoding::UTF8StringEncoding;

    NS::Menu* pMainMenu = NS::Menu::alloc()->init();
    NS::MenuItem* pAppMenuItem = NS::MenuItem::alloc()->init();
    NS::Menu* pAppMenu = NS::Menu::alloc()->init( NS::String::string( "Appname", UTF8StringEncoding ) );

    NS::String* appName = NS::RunningApplication::currentApplication()->localizedName();
    NS::String* quitItemName = NS::String::string( "Quit ", UTF8StringEncoding )->stringByAppendingString( appName );
    SEL quitCb = NS::MenuItem::registerActionCallback( "appQuit", [](void*,SEL,const NS::Object* pSender){
        auto pApp = NS::Application::sharedApplication();
        pApp->terminate( pSender );
    } );

    NS::MenuItem* pAppQuitItem = pAppMenu->addItem( quitItemName, quitCb, NS::String::string( "q", UTF8StringEncoding ) );
    pAppQuitItem->setKeyEquivalentModifierMask( NS::EventModifierFlagCommand );
    pAppMenuItem->setSubmenu( pAppMenu );

    NS::MenuItem* pWindowMenuItem = NS::MenuItem::alloc()->init();
    NS::Menu* pWindowMenu = NS::Menu::alloc()->init( NS::String::string( "Window", UTF8StringEncoding ) );

    SEL closeWindowCb = NS::MenuItem::registerActionCallback( "windowClose", [](void*, SEL, const NS::Object*){
        auto pApp = NS::Application::sharedApplication();
            pApp->windows()->object< NS::Window >(0)->close();
    } );
    NS::MenuItem* pCloseWindowItem = pWindowMenu->addItem( NS::String::string( "Close Window", UTF8StringEncoding ), closeWindowCb, NS::String::string( "w", UTF8StringEncoding ) );
    pCloseWindowItem->setKeyEquivalentModifierMask( NS::EventModifierFlagCommand );

    pWindowMenuItem->setSubmenu( pWindowMenu );

    pMainMenu->addItem( pAppMenuItem );
    pMainMenu->addItem( pWindowMenuItem );

    pAppMenuItem->release();
    pWindowMenuItem->release();
    pAppMenu->release();
    pWindowMenu->release();

    return pMainMenu->autorelease();
}

void MyAppDelegate::applicationWillFinishLaunching( NS::Notification* pNotification )
{
    NS::Menu* pMenu = createMenuBar();
    NS::Application* pApp = reinterpret_cast< NS::Application* >( pNotification->object() );
    pApp->setMainMenu( pMenu );
    pApp->setActivationPolicy( NS::ActivationPolicy::ActivationPolicyRegular );
}

void MyAppDelegate::applicationDidFinishLaunching( NS::Notification* pNotification )
{
    CGRect frame = (CGRect){ {100.0, 100.0}, {float(width), float(height)} };

    _pWindow = NS::Window::alloc()->init(
        frame,
        NS::WindowStyleMaskClosable|NS::WindowStyleMaskTitled,
        NS::BackingStoreBuffered,
        false );
    
    _pDevice = MTL::CreateSystemDefaultDevice();

    _pMtkView = MTK::View::alloc()->init( frame, _pDevice );
    _pMtkView->setColorPixelFormat( MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB );
    _pMtkView->setClearColor( MTL::ClearColor::Make( 1.0, 0.0, 0.0, 1.0 ) );

    _pViewDelegate = new MyMTKViewDelegate( _pDevice );
    _pMtkView->setDelegate( _pViewDelegate );

    _pWindow->setContentView( _pMtkView );
    _pWindow->setTitle( NS::String::string( "01 - Primitive", NS::StringEncoding::UTF8StringEncoding ) );
    
    _pWindow->makeKeyAndOrderFront( nullptr );

    NS::Application* pApp = reinterpret_cast< NS::Application* >( pNotification->object() );
    pApp->activateIgnoringOtherApps( true );
}

bool MyAppDelegate::applicationShouldTerminateAfterLastWindowClosed( NS::Application* pSender )
{
    return true;
}

#pragma endregion AppDelegate }


#pragma mark - ViewDelegate
#pragma region ViewDelegate {

MyMTKViewDelegate::MyMTKViewDelegate( MTL::Device* pDevice )
: MTK::ViewDelegate()
, _pRenderer( new Renderer( pDevice ) )
{
}

MyMTKViewDelegate::~MyMTKViewDelegate()
{
    delete _pRenderer;
}

void MyMTKViewDelegate::drawInMTKView( MTK::View* pView )
{
    _pRenderer->draw( pView );
}

#pragma endregion ViewDelegate }


#pragma mark - Renderer
#pragma region Renderer {

Renderer::Renderer( MTL::Device* pDevice )
: _pDevice( pDevice->retain() )
{
    _pCommandQueue = _pDevice->newCommandQueue();
    buildShaders();
    buildComputePipeline();
    buildTextures();
    buildBuffers();
    
    _semaphore = dispatch_semaphore_create(maxFramesInFlight);
}

Renderer::~Renderer()
{
    _pTexture->release();
    _pBackBuffer->release();
    _pVertexPositionsBuffer->release();
    _pVertexColorsBuffer->release();
    _pIndexBuffer->release();
    for (uint32_t i = 0u; i < maxFramesInFlight; ++i)
    {
        _pUniformBuffer[i]->release();
    }
    _pPSO->release();
    _cPSO->release();
    _pCommandQueue->release();
    _pDevice->release();
}

void Renderer::buildShaders()
{
    using NS::StringEncoding::UTF8StringEncoding;

    const char* shaderSrc = R"(
        #include <metal_stdlib>
        using namespace metal;

        struct v2f
        {
            float4 position [[position]];
            half3 color;
        };
    
        struct UBO
        {
            float2 iResolution;
            float2 iMouse;
            float iTime;
            float iTimeDelta;
        };

        v2f vertex vertexMain( uint vertexId [[vertex_id]],
                               device const float3* positions [[buffer(0)]],
                               device const float3* colors [[buffer(1)]] )
        {
            v2f o;
            o.position = float4( positions[ vertexId ], 1.0 );
            o.color = half3 ( colors[ vertexId ] );
            return o;
        }
    

        half4 fragment fragmentMain( v2f in [[stage_in]],
                                     device const UBO* ubo [[buffer(0)]],
                                     texture2d< half, access::sample > tex [[texture(0)]],
                                     texture2d< half, access::sample > back [[texture(1)]],
                                     texture2d< half, access::write > wri [[texture(2)]])
        {
            const float dt = 0.15;
    
            float2 uv = in.position.xy / (ubo->iResolution);// * 2.0);
            float2 mou = ubo->iMouse / ubo->iResolution;
    
            float s = length( uv - mou )-.01;
            s = smoothstep(0.0, 0.01, s);
            float3 col = mix(float3( ( 0.5 + 0.5 * sin( ubo->iTime ) ) * 0.5, 0.0, ( 0.5 + 0.5 * cos( ubo->iTime ) ) ) * 0.5, float3(0.0, 0.0, 0.0), s);
    
            constexpr sampler samp( address::repeat, filter::linear );
            half4 velocity = tex.sample( samp, uv );
    
            uv -= dt * ubo->iTimeDelta * ( float2( velocity.xy ) );
    
            half4 self = back.sample( samp, uv ) + half4( half3( col ), 0.0 ) * velocity.z;
    
            wri.write( self, uint2( in.position.xy ), 0 );
    
            self = back.sample( samp, in.position.xy / ( ubo->iResolution * 2.0 ) );
    
            return half4( self );
        }
    )";

    NS::Error* pError = nullptr;
    MTL::Library* pLibrary = _pDevice->newLibrary( NS::String::string(shaderSrc, UTF8StringEncoding), nullptr, &pError );
    if ( !pLibrary )
    {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert( false );
    }

    MTL::Function* pVertexFn = pLibrary->newFunction( NS::String::string("vertexMain", UTF8StringEncoding) );
    MTL::Function* pFragFn = pLibrary->newFunction( NS::String::string("fragmentMain", UTF8StringEncoding) );

    MTL::RenderPipelineDescriptor* pDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pDesc->setVertexFunction( pVertexFn );
    pDesc->setFragmentFunction( pFragFn );
    pDesc->colorAttachments()->object(0)->setPixelFormat( MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB );

    _pPSO = _pDevice->newRenderPipelineState( pDesc, &pError );
    if ( !_pPSO )
    {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert( false );
    }

    pVertexFn->release();
    pFragFn->release();
    pDesc->release();
    pLibrary->release();
}

void Renderer::buildComputePipeline()
{
    /*const char* kernel = R"(
        #include <metal_stdlib>
        using namespace metal;
    
        struct cUBO
        {
            float2 iResolution;
            float2 iMouse;
            float iTime;
        };
        
        kernel void FluidSim( texture2d< half, access::write > tex [[texture(0)]],
                              texture2d< half, access::read > backBuffer [[texture(1)]],
                              uint2 index [[thread_position_in_grid]],
                              uint2 gridSize [[threads_per_grid]],
                              device const cUBO* ubo [[buffer(0)]]
                            )
        {
            float2 uv = float2(index) / float2(gridSize);
            float2 mou = ubo->iMouse / ubo->iResolution;
            float s = length(uv-mou)-.04;
            s = smoothstep(0.0, 0.01, s);
            float3 col = mix(float3(1.0, 1.0, 1.0), float3(0.0, 0.0, 0.0), s);
            
            half3 back   = backBuffer.read( index ).rgb;
            half3 left   = backBuffer.read( index + uint2(-1, 0 ) ).rgb;
            half3 right  = backBuffer.read( index + uint2( 1, 0 ) ).rgb;
            half3 top    = backBuffer.read( index + uint2( 0, 1 ) ).rgb;
            half3 bottom = backBuffer.read( index + uint2( 0,-1 ) ).rgb;
    
            half3 result = half3(col) + 1.0 * 0.25 * (left + right + top + bottom) - back;
    
            tex.write(half4(clamp(result, half3(0.0), half3(1.0)), 1.0), index, 0);
        }
    )";*/
    const char* kernel = R"(
        #include <metal_stdlib>
        using namespace metal;
    
        struct cUBO
        {
            float2 iResolution;
            float2 iMouse;
            float2 iVelocity;
            float iTime;
            float iTimeDelta;
        };
        
        kernel void FluidSim( texture2d< half, access::write > tex [[texture(0)]],
                              texture2d< half, access::read > backBuffer [[texture(1)]],
                              uint2 index [[thread_position_in_grid]],
                              uint2 gridSize [[threads_per_grid]],
                              device const cUBO* ubo [[buffer(0)]]
                            )
        {
            float iDx = 0.25;
            float K = 0.2;
            float dt = 0.15;//iDx * 0.5 * 0.5;
            float2 CScale = 1.0 / 2.0;
            float S=K/dt;
    
            float2 uv = float2(index) / float2(gridSize);
            float2 mou = ubo->iMouse / ubo->iResolution;
            float2 p = uv - mou;
            float s = length( p );//-.04;
            float2 sG = p / s;
            s -= 0.001;
            //s = smoothstep(0.0, 0.001, s);
            float3 col = mix(float3(1.0, 1.0, 1.0), float3(0.0, 0.0, 0.0), s);
            
            half4 back   = backBuffer.read( index );
            half3 left   = backBuffer.read( ( index + uint2(-1, 0 ) ) & gridSize ).xyz;
            half3 right  = backBuffer.read( ( index + uint2( 1, 0 ) ) % gridSize ).xyz;
            half3 top    = backBuffer.read( ( index + uint2( 0, 1 ) ) % gridSize ).xyz;
            half3 bottom = backBuffer.read( ( index + uint2( 0,-1 ) ) % gridSize ).xyz;
    
            float3 UdX = float3( right - left ) * iDx;
            float3 UdY = float3( top - bottom ) * iDx;

            float Udiv = UdX.x + UdY.y;

            float2 DdX = float2( UdX.z, UdY.z );
    
            back.z -= ( dt * dot( float3( DdX, Udiv ), float3( back.xyz ) ) );
            back.z = ( clamp( back.z, half( 0.5 ), half( 3.0 ) ) );
    
            float2 PdX = S * DdX;
            float2 Laplacian = float2( left.xy + right.xy + top.xy + bottom.xy - 4.0 * back.xy );
            float2 ViscosityForce = .01 * Laplacian;
    
            uint2 Was = ( index + uint2( -dt * ( back.xy * 0.5 + 0.5 ) * half2( gridSize ) ) );
    
            back.xy = backBuffer.read( Was ).xy;
            float2 ExternalForce = float2(0.0, 0.0);
            float2 velocity = ( mou - ( ubo->iVelocity / ubo->iResolution ) );
            if ( col.x > .0 && length( velocity ) < 0.1 )
            {
                ExternalForce += 50. * velocity;
                //ExternalForce += 0.07 * (ubo->iVelocity / ubo->iResolution);
            }
            back.xy += dt * half2( ViscosityForce + PdX * -1.0 + ExternalForce );

            //if( uv.y < -0.999 || uv.x < -0.999 || uv.x > 0.999 || uv.y > 0.999 )
            //    back.xy *= 0.0;
    
            //tex.write( half4( back ), index, 0 );
            tex.write( half4( half3( col ), 1.0 ), index, 0 );
        }
    )";
    NS::Error* error = nullptr;
    
    MTL::Library* computeLibrary = _pDevice->newLibrary(NS::String::string(kernel, NS::UTF8StringEncoding), nullptr, &error);
    if (!computeLibrary)
    {
        __builtin_printf( "%s", error->localizedDescription()->utf8String() );
        assert(false);
    }
    MTL::Function* FluidSimFn = computeLibrary->newFunction(NS::String::string("FluidSim", NS::UTF8StringEncoding));
    _cPSO = _pDevice->newComputePipelineState(FluidSimFn, &error);
    if (!_cPSO)
    {
        __builtin_printf( "%s", error->localizedDescription()->utf8String() );
        assert(false);
    }
    FluidSimFn->release();
    computeLibrary->release();
}

void Renderer::generateComputeTexture(MTL::CommandBuffer* pCommandBuffer)
{
    assert(pCommandBuffer);
    
    MTL::ComputeCommandEncoder* computeEncoder = pCommandBuffer->computeCommandEncoder();
    computeEncoder->setComputePipelineState(_cPSO);
    computeEncoder->setTexture(_pTexture, 0);
    computeEncoder->setTexture(_pTexture, 1);
    
    reinterpret_cast< cUBO* >( _pComputeUBOBuffer->contents() )->iMouse = _iMouse;
    reinterpret_cast< cUBO* >( _pComputeUBOBuffer->contents() )->iTime = _iTime;
    reinterpret_cast< cUBO* >( _pComputeUBOBuffer->contents() )->iTimeDelta = _iTimeDelta;
    reinterpret_cast< cUBO* >( _pComputeUBOBuffer->contents() )->iVelocity = _iVelocity;
    _pComputeUBOBuffer->didModifyRange( NS::Range::Make( 0, sizeof( cUBO ) ) );
    
    computeEncoder->setBuffer(_pComputeUBOBuffer, 0, 0);
    
    MTL::Size gridSize = MTL::Size(width, height, 1);
    NS::UInteger threadGroupSize = _cPSO->maxTotalThreadsPerThreadgroup();
    //std::cout << "Thread group size: " << (int)threadGroupSize << "\n";
    
    MTL::Size threadGroups = MTL::Size(threadGroupSize, 1, 1);
    
    computeEncoder->dispatchThreads(gridSize, threadGroups);
    computeEncoder->endEncoding();
    //commandBuffer->commit();
}

void Renderer::buildTextures()
{
    MTL::TextureDescriptor* pTextureDesc = MTL::TextureDescriptor::alloc()->init();
    pTextureDesc->setWidth(width);
    pTextureDesc->setHeight(height);
    pTextureDesc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    pTextureDesc->setTextureType(MTL::TextureType2D);
    pTextureDesc->setStorageMode(MTL::StorageModeManaged);
    pTextureDesc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    
    MTL::Texture* texture = _pDevice->newTexture(pTextureDesc);
    _pTexture = texture;
    
    pTextureDesc->release();
    
    MTL::TextureDescriptor* pBackBufferDesc = MTL::TextureDescriptor::alloc()->init();
    pBackBufferDesc->setWidth(width);
    pBackBufferDesc->setHeight(height);
    pBackBufferDesc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    pBackBufferDesc->setTextureType(MTL::TextureType2D);
    pBackBufferDesc->setStorageMode(MTL::StorageModeManaged);
    pBackBufferDesc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    
    MTL::Texture* backBuffer = _pDevice->newTexture(pBackBufferDesc);
    _pBackBuffer = backBuffer;
    
    pBackBufferDesc->release();
}

void Renderer::buildBuffers()
{
    const size_t NumVertices = 4;

    simd::float3 positions[NumVertices] =
    {
        { -1.0f, -1.0f, 0.0f },
        {  1.0f,  1.0f, 0.0f },
        { -1.0f,  1.0f, 0.0f },
        {  1.0f, -1.0f, 0.0f }
    };

    simd::float3 colors[NumVertices] =
    {
        {  1.0,  0.3f, 0.2f },
        {  0.8f, 1.0f, 0.0f },
        {  0.8f, 0.0f, 1.0f },
        {  1.0f, 0.0f, 1.0f }
    };
    
    uint16_t indices[] = {
        2, 1, 0,
        0, 3, 1
    };

    UBO ubo;
    ubo.iMouse = { 0.0f, 0.0f };
    ubo.iResolution = { float(width), float(height) };
    ubo.iTime = 0.0f;
    
    cUBO cUbo;
    cUbo.iMouse = { 0.0f, 0.0f };
    cUbo.iResolution = { float(width), float(height) };
    cUbo.iTime = 0.0f;
        
    const size_t positionsDataSize = NumVertices * sizeof( simd::float3 );
    const size_t colorDataSize = NumVertices * sizeof( simd::float3 );
    const size_t indexDataSize = sizeof( indices );
    const size_t uniformDataSize = sizeof( ubo );
    const size_t cUniformDataSize = sizeof( cUbo );
    
    MTL::Buffer* pVertexPositionsBuffer = _pDevice->newBuffer( positionsDataSize, MTL::ResourceStorageModeManaged );
    MTL::Buffer* pVertexColorsBuffer = _pDevice->newBuffer( colorDataSize, MTL::ResourceStorageModeManaged );
    MTL::Buffer* pIndexBuffer = _pDevice->newBuffer( indexDataSize, MTL::ResourceStorageModeManaged );
    for (uint32_t i = 0u; i < maxFramesInFlight; ++i)
    {
        MTL::Buffer* pUniformBuffer = _pDevice->newBuffer( uniformDataSize, MTL::ResourceStorageModeManaged );
        _pUniformBuffer[i] = pUniformBuffer;
        memcpy( _pUniformBuffer[i]->contents(), &ubo, uniformDataSize );
        _pUniformBuffer[i]->didModifyRange( NS::Range::Make( 0, _pUniformBuffer[i]->length() ) );
    }
    MTL::Buffer* pCUniformBuffer = _pDevice->newBuffer( cUniformDataSize, MTL::ResourceStorageModeManaged );
    _pComputeUBOBuffer = pCUniformBuffer;
    _pVertexPositionsBuffer = pVertexPositionsBuffer;
    _pVertexColorsBuffer = pVertexColorsBuffer;
    _pIndexBuffer = pIndexBuffer;
    //_pUniformBuffer = pUniformBuffer;

    memcpy( _pVertexPositionsBuffer->contents(), positions, positionsDataSize );
    memcpy( _pVertexColorsBuffer->contents(), colors, colorDataSize );
    memcpy( _pIndexBuffer->contents(), indices, indexDataSize );
    memcpy( _pComputeUBOBuffer->contents(), &cUbo, cUniformDataSize );
    //memcpy( _pUniformBuffer->contents(), &ubo, uniformDataSize );

    _pVertexPositionsBuffer->didModifyRange( NS::Range::Make( 0, _pVertexPositionsBuffer->length() ) );
    _pVertexColorsBuffer->didModifyRange( NS::Range::Make( 0, _pVertexColorsBuffer->length() ) );
    _pIndexBuffer->didModifyRange( NS::Range::Make( 0, _pIndexBuffer->length() ) );
    _pComputeUBOBuffer->didModifyRange( NS::Range::Make( 0, _pComputeUBOBuffer->length() ) );
    //_pUniformBuffer->didModifyRange( NS::Range::Make( 0, _pUniformBuffer->length() ) );
}

void Renderer::draw( MTK::View* pView )
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    MTL::CommandBuffer* pCmd = _pCommandQueue->commandBuffer();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    
    Renderer* pRenderer = this;
    pCmd->addCompletedHandler( ^void( MTL::CommandBuffer* pCmd) {
        dispatch_semaphore_signal( pRenderer->_semaphore );
    });
    
    // Calculate the time between frames.
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    _iTime = std::chrono::duration<float, std::chrono::seconds::period>( currentTime - startTime ).count();

    _iTimeDelta = _iTime - _iLastTime;
    _iLastTime = _iTime;
    
    CGPoint cursor = CGEventGetLocation(CGEventCreate(NULL));
    _iMouse = { float(cursor.x), float(cursor.y) };
    printf("Mouse pos: (%f, %f)\n", _iVelocity.x, _iVelocity.y);
    
    _frame = (_frame + 1) % maxFramesInFlight;
    
    reinterpret_cast< UBO* >( _pUniformBuffer[_frame]->contents() )->iMouse = _iMouse;
    reinterpret_cast< UBO* >( _pUniformBuffer[_frame]->contents() )->iTime = _iTime;
    reinterpret_cast< UBO* >( _pUniformBuffer[_frame]->contents() )->iTimeDelta = _iTimeDelta;
    _pUniformBuffer[_frame]->didModifyRange( NS::Range::Make( 0, sizeof( UBO ) ) );
    
    generateComputeTexture( pCmd );
    
    _iVelocity = _iMouseLast;
    //_iVelocity.x /= float( width );
    //_iVelocity.y /= float( height );
    _iMouseLast = _iMouse;
    
    std::cout << _iVelocity.x << ": X\n";
    std::cout << _iVelocity.x << ": X\n";
    
    MTL::RenderPassDescriptor* pRpd = pView->currentRenderPassDescriptor();
    MTL::RenderCommandEncoder* pEnc = pCmd->renderCommandEncoder( pRpd );

    pEnc->setRenderPipelineState( _pPSO );
    pEnc->setVertexBuffer( _pVertexPositionsBuffer, 0, 0 );
    pEnc->setVertexBuffer( _pVertexColorsBuffer, 0, 1 );
    pEnc->setFragmentBuffer( _pUniformBuffer[_frame], 0, 0 );
    pEnc->setFragmentTexture( _pTexture, 0 );
    pEnc->setFragmentTexture( _pBackBuffer, 1 );
    pEnc->setFragmentTexture( _pBackBuffer, 2 );
    
    pEnc->drawIndexedPrimitives( MTL::PrimitiveType::PrimitiveTypeTriangle,
                                6, MTL::IndexType::IndexTypeUInt16,
                                _pIndexBuffer,
                                0,
                                1 );
    //pEnc->drawPrimitives( MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(4) );

    pEnc->endEncoding();
    pCmd->presentDrawable( pView->currentDrawable() );
    pCmd->commit();

    pPool->release();
}

#pragma endregion Renderer }
