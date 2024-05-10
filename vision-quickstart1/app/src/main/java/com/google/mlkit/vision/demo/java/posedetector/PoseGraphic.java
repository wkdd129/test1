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

import static java.lang.Math.max;
import static java.lang.Math.min;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.GraphicOverlay.Graphic;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseLandmark;
import java.util.List;
import java.util.Locale;

/** Draw the detected pose in preview. */
public class PoseGraphic extends Graphic {
  private int repCount;
  private static final float DOT_RADIUS = 8.0f;
  private static final float IN_FRAME_LIKELIHOOD_TEXT_SIZE = 30.0f;
  private static final float STROKE_WIDTH = 10.0f;
  private static final float POSE_CLASSIFICATION_TEXT_SIZE = 60.0f;

  private final Pose pose;
  private final boolean showInFrameLikelihood;
  private final boolean visualizeZ;
  private final boolean rescaleZForVisualization;
  private float zMin = Float.MAX_VALUE;
  private float zMax = Float.MIN_VALUE;

  private final List<String> poseClassification;
  private final Paint classificationTextPaint;
  private final Paint leftPaint;
  private final Paint rightPaint;
  private final Paint whitePaint;

  PoseGraphic(
          GraphicOverlay overlay,
          Pose pose,
          boolean showInFrameLikelihood,
          boolean visualizeZ,
          boolean rescaleZForVisualization,
          List<String> poseClassification) {

    super(overlay);
    this.repCount = repCount;
    this.pose = pose;
    this.showInFrameLikelihood = showInFrameLikelihood;
    this.visualizeZ = visualizeZ;
    this.rescaleZForVisualization = rescaleZForVisualization;

    this.poseClassification = poseClassification;
    classificationTextPaint = new Paint();
    classificationTextPaint.setColor(Color.WHITE);
    classificationTextPaint.setTextSize(POSE_CLASSIFICATION_TEXT_SIZE);
    classificationTextPaint.setShadowLayer(5.0f, 0f, 0f, Color.BLACK);

    whitePaint = new Paint();
    whitePaint.setStrokeWidth(STROKE_WIDTH);
    whitePaint.setColor(Color.WHITE);
    whitePaint.setTextSize(IN_FRAME_LIKELIHOOD_TEXT_SIZE);
    leftPaint = new Paint();
    leftPaint.setStrokeWidth(STROKE_WIDTH);
    leftPaint.setColor(Color.GREEN);
    rightPaint = new Paint();
    rightPaint.setStrokeWidth(STROKE_WIDTH);
    rightPaint.setColor(Color.YELLOW);
  }

  @Override
  public void draw(Canvas canvas) {
    List<PoseLandmark> landmarks = pose.getAllPoseLandmarks();
    if (landmarks.isEmpty()) {
      return;
    }

    // Draw pose classification text.
    float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
    for (int i = 0; i < poseClassification.size(); i++) {
      float classificationY =
              (canvas.getHeight()
                      - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f * (poseClassification.size() - i));
      canvas.drawText(
              poseClassification.get(i), classificationX, classificationY, classificationTextPaint);
    }

    // Draw all the points
    for (PoseLandmark landmark : landmarks) {
      drawPoint(canvas, landmark, whitePaint);
      if (visualizeZ && rescaleZForVisualization) {
        zMin = min(zMin, landmark.getPosition3D().getZ());
        zMax = max(zMax, landmark.getPosition3D().getZ());
      }
    }

    PoseLandmark nose = pose.getPoseLandmark(PoseLandmark.NOSE);
    PoseLandmark lefyEyeInner = pose.getPoseLandmark(PoseLandmark.LEFT_EYE_INNER);
    PoseLandmark lefyEye = pose.getPoseLandmark(PoseLandmark.LEFT_EYE);
    PoseLandmark leftEyeOuter = pose.getPoseLandmark(PoseLandmark.LEFT_EYE_OUTER);
    PoseLandmark rightEyeInner = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE_INNER);
    PoseLandmark rightEye = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE);
    PoseLandmark rightEyeOuter = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE_OUTER);
    PoseLandmark leftEar = pose.getPoseLandmark(PoseLandmark.LEFT_EAR);
    PoseLandmark rightEar = pose.getPoseLandmark(PoseLandmark.RIGHT_EAR);
    PoseLandmark leftMouth = pose.getPoseLandmark(PoseLandmark.LEFT_MOUTH);
    PoseLandmark rightMouth = pose.getPoseLandmark(PoseLandmark.RIGHT_MOUTH);

    PoseLandmark leftShoulder = pose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER);
    PoseLandmark rightShoulder = pose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER);
    PoseLandmark leftElbow = pose.getPoseLandmark(PoseLandmark.LEFT_ELBOW);
    PoseLandmark rightElbow = pose.getPoseLandmark(PoseLandmark.RIGHT_ELBOW);
    PoseLandmark leftWrist = pose.getPoseLandmark(PoseLandmark.LEFT_WRIST);
    PoseLandmark rightWrist = pose.getPoseLandmark(PoseLandmark.RIGHT_WRIST);
    PoseLandmark leftHip = pose.getPoseLandmark(PoseLandmark.LEFT_HIP);
    PoseLandmark rightHip = pose.getPoseLandmark(PoseLandmark.RIGHT_HIP);
    PoseLandmark leftKnee = pose.getPoseLandmark(PoseLandmark.LEFT_KNEE);
    PoseLandmark rightKnee = pose.getPoseLandmark(PoseLandmark.RIGHT_KNEE);
    PoseLandmark leftAnkle = pose.getPoseLandmark(PoseLandmark.LEFT_ANKLE);
    PoseLandmark rightAnkle = pose.getPoseLandmark(PoseLandmark.RIGHT_ANKLE);

    PoseLandmark leftPinky = pose.getPoseLandmark(PoseLandmark.LEFT_PINKY);
    PoseLandmark rightPinky = pose.getPoseLandmark(PoseLandmark.RIGHT_PINKY);
    PoseLandmark leftIndex = pose.getPoseLandmark(PoseLandmark.LEFT_INDEX);
    PoseLandmark rightIndex = pose.getPoseLandmark(PoseLandmark.RIGHT_INDEX);
    PoseLandmark leftThumb = pose.getPoseLandmark(PoseLandmark.LEFT_THUMB);
    PoseLandmark rightThumb = pose.getPoseLandmark(PoseLandmark.RIGHT_THUMB);
    PoseLandmark leftHeel = pose.getPoseLandmark(PoseLandmark.LEFT_HEEL);
    PoseLandmark rightHeel = pose.getPoseLandmark(PoseLandmark.RIGHT_HEEL);
    PoseLandmark leftFootIndex = pose.getPoseLandmark(PoseLandmark.LEFT_FOOT_INDEX);
    PoseLandmark rightFootIndex = pose.getPoseLandmark(PoseLandmark.RIGHT_FOOT_INDEX);

    // Face
    drawLine(canvas, nose, lefyEyeInner, whitePaint);
    drawLine(canvas, lefyEyeInner, lefyEye, whitePaint);
    drawLine(canvas, lefyEye, leftEyeOuter, whitePaint);
    drawLine(canvas, leftEyeOuter, leftEar, whitePaint);
    drawLine(canvas, nose, rightEyeInner, whitePaint);
    drawLine(canvas, rightEyeInner, rightEye, whitePaint);
    drawLine(canvas, rightEye, rightEyeOuter, whitePaint);
    drawLine(canvas, rightEyeOuter, rightEar, whitePaint);
    drawLine(canvas, leftMouth, rightMouth, whitePaint);

    drawLine(canvas, leftShoulder, rightShoulder, whitePaint);
    drawLine(canvas, leftHip, rightHip, whitePaint);

    // Left body
    drawLine(canvas, leftShoulder, leftElbow, leftPaint);
    drawLine(canvas, leftElbow, leftWrist, leftPaint);
    drawLine(canvas, leftShoulder, leftHip, leftPaint);
    drawLine(canvas, leftHip, leftKnee, leftPaint);
    drawLine(canvas, leftKnee, leftAnkle, leftPaint);
    drawLine(canvas, leftWrist, leftThumb, leftPaint);
    drawLine(canvas, leftWrist, leftPinky, leftPaint);
    drawLine(canvas, leftWrist, leftIndex, leftPaint);
    drawLine(canvas, leftIndex, leftPinky, leftPaint);
    drawLine(canvas, leftAnkle, leftHeel, leftPaint);
    drawLine(canvas, leftHeel, leftFootIndex, leftPaint);

    // Right body
    drawLine(canvas, rightShoulder, rightElbow, rightPaint);
    drawLine(canvas, rightElbow, rightWrist, rightPaint);
    drawLine(canvas, rightShoulder, rightHip, rightPaint);
    drawLine(canvas, rightHip, rightKnee, rightPaint);
    drawLine(canvas, rightKnee, rightAnkle, rightPaint);
    drawLine(canvas, rightWrist, rightThumb, rightPaint);
    drawLine(canvas, rightWrist, rightPinky, rightPaint);
    drawLine(canvas, rightWrist, rightIndex, rightPaint);
    drawLine(canvas, rightIndex, rightPinky, rightPaint);
    drawLine(canvas, rightAnkle, rightHeel, rightPaint);
    drawLine(canvas, rightHeel, rightFootIndex, rightPaint);

    // Draw inFrameLikelihood for all points
    if (showInFrameLikelihood) {
      for (PoseLandmark landmark : landmarks) {
        canvas.drawText(
                String.format(Locale.US, "%.2f", landmark.getInFrameLikelihood()),
                translateX(landmark.getPosition().x),
                translateY(landmark.getPosition().y),
                whitePaint);
      }
    }
  }

  void drawPoint(Canvas canvas, PoseLandmark landmark, Paint paint) {
    PointF3D point = landmark.getPosition3D();
    updatePaintColorByZValue(
            paint, canvas, visualizeZ, rescaleZForVisualization, point.getZ(), zMin, zMax);
    canvas.drawCircle(translateX(point.getX()), translateY(point.getY()), DOT_RADIUS, paint);
  }

  void drawLine(Canvas canvas, PoseLandmark startLandmark, PoseLandmark endLandmark, Paint paint) {
    PointF3D start = startLandmark.getPosition3D();
    PointF3D end = endLandmark.getPosition3D();

    // Gets average z for the current body line
    float avgZInImagePixel = (start.getZ() + end.getZ()) / 2;
    updatePaintColorByZValue(
            paint, canvas, visualizeZ, rescaleZForVisualization, avgZInImagePixel, zMin, zMax);

    canvas.drawLine(
            translateX(start.getX()),
            translateY(start.getY()),
            translateX(end.getX()),
            translateY(end.getY()),
            paint);
  }
}

//代码解析: PoseGraphic 类用于绘制检测到的姿势
//这段 Java 代码定义了一个名为 PoseGraphic 的类，它继承自 GraphicOverlay.Graphic 类，用于在预览界面上绘制检测到的人体姿势。它利用 ML Kit 的
// Pose Detection API 提供的关键点信息，将人体骨骼结构和关键点可视化。
//主要功能:
//绘制姿势关键点: 将检测到的关键点绘制成圆点，并可选择显示其置信度值。
//绘制骨骼线条: 连接相邻的关键点，形成人体骨骼结构的线条。
//可视化 Z 轴信息 (可选): 根据关键点的 Z 轴坐标值，调整绘制的颜色，以显示深度信息。
//显示分类结果: 将姿势分类结果绘制成文本，显示在图像上方。
//代码解读:
//构造函数: 接受姿势对象 (Pose)、是否显示置信度 (showInFrameLikelihood)、是否可视化 Z 轴 (visualizeZ)、是否重新缩放 Z 轴 (rescaleZForVisualization) 以及姿势分类结果 (poseClassification) 等参数。
//draw(Canvas canvas) 方法:
//获取所有关键点坐标。
//绘制姿势分类结果文本。
//遍历所有关键点，并调用 drawPoint 方法绘制关键点圆点。
//获取特定关键点，例如鼻子、眼睛、肩膀等。
//使用 drawLine 方法绘制连接关键点的骨骼线条，例如头部、躯干和四肢。
//如果 showInFrameLikelihood 为 true，则在每个关键点旁边绘制其置信度值。
//drawPoint(Canvas canvas, PoseLandmark landmark, Paint paint) 方法:
//获取关键点的三维坐标。
//根据 Z 轴坐标值更新绘制颜色 (如果 visualizeZ 和 rescaleZForVisualization 为 true)。
//绘制关键点圆点。
//drawLine(Canvas canvas, PoseLandmark startLandmark, PoseLandmark endLandmark, Paint paint) 方法:
//获取起始关键点和结束关键点的三维坐标。
//计算两点之间 Z 轴坐标的平均值。
//根据平均 Z 轴坐标值更新绘制颜色 (如果 visualizeZ 和 rescaleZForVisualization 为 true)。
//绘制连接两点的骨骼线条。
//代码应用:
//这个代码可以用于各种需要可视化人体姿势的应用，例如：
//健身应用: 显示用户的健身动作，并提供实时反馈。
//运动分析: 分析运动员的动作，并可视化关键点和骨骼结构。
//康复训练: 帮助患者进行康复训练，并可视化其动作。
//总结:
//PoseGraphic 类提供了一种有效的方法，用于可视化人体姿势的关键点和骨骼结构。它可以与 ML Kit 的 Pose Detection API 结合使用，方便地将检测到的姿势显示在应用程序界面上。