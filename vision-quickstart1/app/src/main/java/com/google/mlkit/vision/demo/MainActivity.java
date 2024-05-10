package com.google.mlkit.vision.demo;

import android.Manifest;
import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.MediaRecorder;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.IBinder;
import android.util.DisplayMetrics;
import android.view.View;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class
MainActivity extends Activity {

    private ScreenRecordService screenRecordService;
    private MediaProjectionManager mediaProjectionManager;
    private MediaProjection mediaProjection;
    private DisplayMetrics metrics;
    private static final int RECORD_REQUEST_CODE = 101;



    private static final int PERMISSION_CODE = 1;
    private MediaRecorder mediaRecorder;
    private VirtualDisplay virtualDisplay;
    private static final int SCREEN_CAPTURE_REQUEST_CODE = 1000;
    private static final int DISPLAY_WIDTH = 720;
    private static final int DISPLAY_HEIGHT = 1280;
    private static final int SCREEN_DENSITY = 320;
    private static final String[] PERMISSIONS = {
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.RECORD_AUDIO
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 初始化 MediaProjectionManager
        mediaProjectionManager = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);

        // 设置“开始录屏”按钮的点击事件
        findViewById(R.id.start_record_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startScreenCapture();
            }
        });

        // 设置“停止录屏”按钮的点击事件
        findViewById(R.id.stop_record_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                stopScreenRecording();
            }
        });
    }

    // 请求权限并启动屏幕录制
    private void startScreenCapture() {
        Intent captureIntent = mediaProjectionManager.createScreenCaptureIntent();
        startActivityForResult(captureIntent, SCREEN_CAPTURE_REQUEST_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == SCREEN_CAPTURE_REQUEST_CODE) {
            if (resultCode == Activity.RESULT_OK && data != null) {
                try {
                    mediaProjection = mediaProjectionManager.getMediaProjection(resultCode, data);
                    setUpMediaRecorder(); // 配置 MediaRecorder
                    createVirtualDisplay(); // 创建 VirtualDisplay
                    mediaRecorder.start(); // 开始录制
                    Toast.makeText(this, "屏幕录制已开始", Toast.LENGTH_SHORT).show();
                } catch (Exception e) {
                    e.printStackTrace();
                    Toast.makeText(this, "屏幕录制初始化失败：" + e.getMessage(), Toast.LENGTH_LONG).show();
                }
            } else {
                Toast.makeText(this, "录屏请求被拒绝", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // 设置 MediaRecorder 参数
    private void setUpMediaRecorder() {
        try {
            mediaRecorder = new MediaRecorder();
            mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
            mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            mediaRecorder.setOutputFile("/ScreenRecording.mp4");
            mediaRecorder.setVideoSize(DISPLAY_WIDTH, DISPLAY_HEIGHT);
            mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
            mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            mediaRecorder.setVideoFrameRate(30);
            mediaRecorder.setVideoEncodingBitRate(5 * 1024 * 1024);
            mediaRecorder = new MediaRecorder();
            //设置声音来源
            mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            //设置视频来源
            mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
            //设置视频格式
            mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            //设置视频储存地址
            //设置视频编码
            mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
            //设置声音编码
            mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
            //视频码率
            mediaRecorder.setVideoEncodingBitRate(2 * 1920 * 1080);
            mediaRecorder.setVideoFrameRate(18);
            mediaRecorder.prepare();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // 创建 VirtualDisplay 用于录制屏幕
    private void createVirtualDisplay() {
        virtualDisplay = mediaProjection.createVirtualDisplay("ScreenRecording",
                DISPLAY_WIDTH, DISPLAY_HEIGHT, SCREEN_DENSITY,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                mediaRecorder.getSurface(), null, null);
    }

    // 停止屏幕录制
    private void stopScreenRecording() {
        if (mediaRecorder != null) {
            mediaRecorder.stop();
            mediaRecorder.reset();
            mediaRecorder = null;
        }
        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }
        if (mediaProjection != null) {
            mediaProjection.stop();
            mediaProjection = null;
        }
        Toast.makeText(this, "屏幕录制已停止", Toast.LENGTH_SHORT).show();
    }
}
