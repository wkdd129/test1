package com.google.mlkit.vision.demo;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.MediaRecorder;
import android.media.projection.MediaProjection;
import android.os.Binder;
import android.os.Build;
import android.os.Environment;
import android.os.HandlerThread;
import android.os.IBinder;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;

import androidx.core.app.NotificationCompat;

public class ScreenRecordService extends Service {

    private MediaProjection mediaProjection;
    private MediaRecorder mediaRecorder;
    private VirtualDisplay virtualDisplay;

    private boolean running;
    private int width = 720;
    private int height = 1080;
    private int dpi;

    private String videoPath = "";

    private static final String NOTIFICATION_CHANNEL_ID = "com.hhw.screencapture.id";
    private static final String NOTIFICATION_CHANNEL_NAME = "com.hhw.screencapture.name";

    @Override
    public void onCreate() {
        super.onCreate();
        notification();
        HandlerThread serviceThread = new HandlerThread("service_thread", android.os.Process.THREAD_PRIORITY_BACKGROUND);
        serviceThread.start();
        running = false;
    }

    public class ScreenRecordBinder extends Binder {
        public ScreenRecordService getScreenRecordService(){
            return ScreenRecordService.this;
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return new ScreenRecordBinder();
    }


    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }

    @Override
    public boolean onUnbind(Intent intent) {
        return super.onUnbind(intent);
    }

    public boolean isRunning() {
        return running;
    }

    /**
     * 需要在开始录屏前调用
     */
    public void setMediaProject(MediaProjection project) {
        mediaProjection = project;
    }

    /**
     * 设置录制参数，需要在开始录像前调用
     *
     * @param width 录制分辨率宽
     * @param height 录制分辨率高
     * @param dpi 录制密度
     */
    public void setConfig(int width, int height, int dpi) {
        this.width = width;
        this.height = height;
        this.dpi = dpi;
    }

    /**
     * 开始录屏
     * @return
     */
    public boolean startRecord() {
        if (mediaProjection == null || running) {
            return false;
        }
        initRecorder();
        createVirtualDisplay();
        try {
            mediaRecorder.start();
            running = true;
            return true;
        }catch (IllegalStateException e){
            e.printStackTrace();
            Toast.makeText(this,"start 出错，录屏失败！",Toast.LENGTH_SHORT).show();
            running = false;
            return false;
        }
    }

    /**
     *
     * @return
     */
    public boolean stopRecord() {
        if (!running) {
            return false;
        }
        running = false;
        try{
            mediaRecorder.stop();
            mediaRecorder.reset();
            virtualDisplay.release();
        }catch (Exception e){
            e.printStackTrace();
            Toast.makeText(ScreenRecordService.this, "录屏出错,保存失败", Toast.LENGTH_SHORT).show();
            return false;
        }
        Toast.makeText(ScreenRecordService.this, "录屏完成，已保存。", Toast.LENGTH_SHORT).show();
        return true;
    }

    private void createVirtualDisplay() {
        try{
            virtualDisplay = mediaProjection.createVirtualDisplay("MainScreen", width, height, dpi,
                    DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, mediaRecorder.getSurface(), null, null);
            //将MediaRecorder和MediaProjection绑定
        }catch (Exception e){
            Toast.makeText(this,"virtualDisplay 录屏出错！",Toast.LENGTH_SHORT).show();
        }
    }

    private void initRecorder() {
        mediaRecorder = new MediaRecorder();
        //设置声音来源
        mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        //设置视频来源
        mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        //设置视频格式
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        //设置视频储存地址
        videoPath = getSaveDirectory() + System.currentTimeMillis() + ".mp4";
        mediaRecorder.setOutputFile(videoPath);
        //设置视频大小
        mediaRecorder.setVideoSize(width, height);
        //设置视频编码
        mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        //设置声音编码
        mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
        //视频码率
        mediaRecorder.setVideoEncodingBitRate(2 * 1920 * 1080);
        mediaRecorder.setVideoFrameRate(18);
        try {
            mediaRecorder.prepare();
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this,"prepare出错，录屏失败！",Toast.LENGTH_SHORT).show();
        }
    }

    public String getSaveDirectory() {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            String rootDir = Environment.getExternalStorageDirectory()
                    .getAbsolutePath() + "/" + "手机录屏助手/" + "/";
            File file = new File(rootDir);
            if (!file.exists()) {
                if (!file.mkdirs()) {
                    return null;
                }
            }
            return rootDir;
        } else {
            return null;
        }
    }


    public void notification() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Intent notificationIntent = new Intent(this, ScreenRecordService.class);
            PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, notificationIntent, 0);
            NotificationCompat.Builder notificationBuilder = new NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
                    .setSmallIcon(R.drawable.ic_launcher_foreground)
                    .setContentTitle("Starting Service")
                    .setContentText("录屏服务运行中")
                    .setContentIntent(pendingIntent);
            Notification notification = notificationBuilder.build();
            NotificationChannel channel = new NotificationChannel(NOTIFICATION_CHANNEL_ID, NOTIFICATION_CHANNEL_NAME, NotificationManager.IMPORTANCE_DEFAULT);
            NotificationManager notificationManager = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
            notificationManager.createNotificationChannel(channel);
            startForeground(10, notification);
            ////第一个参数不能为0，且必须使用此方法显示通知，不能使用notificationManager.notify，否则还是会报错误
        }
    }

}
