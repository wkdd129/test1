/*
 * Copyright 2020 Google LLC. All rights reserved.
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

package com.google.mlkit.vision.demo.java.posedetector;

import android.content.Context;
import android.media.MediaRecorder;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Button;

import androidx.annotation.NonNull;
import com.google.android.gms.tasks.Task;
import com.google.android.odml.image.MlImage;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.java.VisionProcessorBase;
import com.google.mlkit.vision.demo.java.posedetector.classification.PoseClassifierProcessor;

import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseDetector;
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/** A processor to run pose detector. */
public class PoseDetectorProcessor
        extends VisionProcessorBase<PoseDetectorProcessor.PoseWithClassification> {
  private static final String TAG = "PoseDetectorProcessor";

  private enum PoseState {
    DOWN,
    UP,
    T1,
    T2
  }
  private final PoseDetector detector;

  private SurfaceView surfaceView;

  private final boolean showInFrameLikelihood;
  private final boolean visualizeZ;
  private final boolean rescaleZForVisualization;
  private final boolean runClassification;
  private final boolean isStreamMode;
  private final Context context;
  private final Executor classificationExecutor;
  private PoseState currentState = PoseState.DOWN; // 初始状态
  private PoseClassifierProcessor poseClassifierProcessor;
  /** Internal class to hold Pose and classification results. */
  protected static class PoseWithClassification {
    private final Pose pose;
    private final List<String> classificationResult;

    public PoseWithClassification(Pose pose, List<String> classificationResult) {
      this.pose = pose;
      this.classificationResult = classificationResult;
    }

    public Pose getPose() {
      return pose;
    }

    public List<String> getClassificationResult() {
      return classificationResult;
    }
  }

  public PoseDetectorProcessor(
          Context context,
          PoseDetectorOptionsBase options,
          boolean showInFrameLikelihood,
          boolean visualizeZ,
          boolean rescaleZForVisualization,
          boolean runClassification,
          boolean isStreamMode) {
    super(context);
    this.showInFrameLikelihood = showInFrameLikelihood;
    this.visualizeZ = visualizeZ;
    this.rescaleZForVisualization = rescaleZForVisualization;
    detector = PoseDetection.getClient(options);
    this.runClassification = runClassification;
    this.isStreamMode = isStreamMode;
    this.context = context;
    classificationExecutor = Executors.newSingleThreadExecutor();


  }

  @Override
  public void stop() {
    super.stop();
    detector.close();
  }



  @Override
  protected Task<PoseWithClassification> detectInImage(InputImage image) {
    return detector
            .process(image)
            .continueWith(
                    classificationExecutor,
                    task -> {
                      Pose pose = task.getResult();
                      List<String> classificationResult = new ArrayList<>();
                      if (runClassification) {
                        if (poseClassifierProcessor == null) {
                          poseClassifierProcessor = new PoseClassifierProcessor(context, isStreamMode);
                        }
                        classificationResult = poseClassifierProcessor.getPoseResult(pose);
                      }
                      return new PoseWithClassification(pose, classificationResult);
                    });
  }

  @Override
  protected Task<PoseWithClassification> detectInImage(MlImage image) {
    return detector
            .process(image)
            .continueWith(
                    classificationExecutor,
                    task -> {
                      Pose pose = task.getResult();
                      List<String> classificationResult = new ArrayList<>();
                      if (runClassification) {
                        if (poseClassifierProcessor == null) {
                          poseClassifierProcessor = new PoseClassifierProcessor(context, isStreamMode);
                        }
                        classificationResult = poseClassifierProcessor.getPoseResult(pose);
                      }
                      return new PoseWithClassification(pose, classificationResult);
                    });
  }

  @Override
  protected void onSuccess(
          @NonNull PoseWithClassification poseWithClassification,
          @NonNull GraphicOverlay graphicOverlay) {

    graphicOverlay.add(

            new PoseGraphic(
                    graphicOverlay,
                    poseWithClassification.pose,
                    showInFrameLikelihood,
                    visualizeZ,
                    rescaleZForVisualization,
                    poseWithClassification.classificationResult
            ));
  }

  @Override
  protected void onFailure(@NonNull Exception e) {
    Log.e(TAG, "Pose detection failed!", e);
  }


  @Override
  protected boolean isMlImageEnabled(Context context) {
    // Use MlImage in Pose Detection by default, change it to OFF to switch to InputImage.
    return true;
  }





}
//代码解析：PoseDetectorProcessor 类用于处理姿势检测并进行分类
//这个 Java 代码定义了一个名为 PoseDetectorProcessor 姿势检测处理器 的类，它继承自 VisionProcessorBase 视觉处理器库 类，
// 用于处理姿势检测任务，并可选地进行姿势分类。它利用 ML Kit 的 Pose Detection API 检测图像或视频帧中的人体姿势，并可视化检测结果。
// 此外，它还可以与 PoseClassifierProcessor 姿势分类处理器 类结合，对检测到的姿势进行分类，并计算重复动作的次数。
//主要功能：
//姿势检测： 使用 ML Kit 的 Pose Detection API 检测图像或视频帧中的人体姿势。
//姿势可视化： 将检测到的姿势关键点绘制到 GraphicOverlay 图形叠加 上。
//姿势分类 (可选): 使用 PoseClassifierProcessor 姿势分类处理器 类对检测到的姿势进行分类，并计算重复动作次数。
//结果输出： 将分类结果和重复次数显示在界面上。
//代码解读：
//类成员变量:
//detector 探测器：ML Kit 的 Pose Detector 对象。
//showInFrameLikelihood：是否显示关键点的置信度。
//visualizeZ：是否可视化 Z 轴坐标。
//rescaleZForVisualization 重新缩放ZFor可视化：是否重新缩放 Z 轴坐标以进行可视化。
//runClassification：是否进行姿势分类。
//isStreamMode：是否处于流模式，即处理连续的视频帧。
//context：应用程序上下文。
//classificationExecutor：用于执行分类任务的线程池。
//currentState：当前姿势状态。
//sharedViewModel：用于在 UI 组件之间共享数据的 ViewModel 对象。
//poseClassifierProcessor：用于姿势分类的 PoseClassifierProcessor 姿势分类处理器 对象。
//PoseWithClassification 分类姿势 类： 一个内部类，用于存储检测到的姿势和分类结果。
//构造函数： 初始化成员变量，并创建 PoseDetector 姿势检测器 对象。
//stop() 方法： 停止姿势检测器。
//detectInImage(InputImage image)
//检测图像（输入图像图像） 和 detectInImage(MlImage image)
//检测InImage（MlImage图像） 方法：
//使用 PoseDetector 检测图像中的姿势。
//如果 runClassification 运行分类 为 true，则使用 PoseClassifierProcessor 姿势分类处理器 对检测到的姿势进行分类。
//将姿势和分类结果封装到 PoseWithClassification 分类姿势 对象中，并返回。
//onSuccess(poseWithClassification, graphicOverlay)
//onSuccess（poseWithClassification，graphicOverlay） 方法：
//创建 PoseGraphic 对象，并将姿势和分类结果传递给它。
//将 PoseGraphic 对象添加到 GraphicOverlay 图形叠加 上，进行可视化。
//onFailure(e) 失败时(e) 方法： 记录姿势检测失败的错误信息。
//isMlImageEnabled(context)
//isMlImageEnabled（上下文） 方法： 指定是否使用 MlImage 图像处理 作为输入，默认为 true。
//代码应用：
//这个代码可以用于各种需要进行姿势检测和分类的应用，例如：
//健身应用: 识别用户的健身动作，并提供实时反馈。
//运动分析: 分析运动员的动作，提供训练建议。
//康复训练: 帮助患者进行康复训练，并监测训练效果。
//总结：
//PoseDetectorProcessor 类是一个功能强大的工具，可以用于检测和分类人体姿势。它结合了 ML Kit 的 Pose Detection API 和自定义的姿势分类器，可以实现多种姿势识别和分析应用。