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

package com.google.mlkit.vision.demo.java.posedetector.classification;

import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.average;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.l2Norm2D;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.multiplyAll;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.subtract;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.subtractAll;

import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.pose.PoseLandmark;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates embedding for given list of Pose landmarks.
 */
public class PoseEmbedding {
  // Multiplier to apply to the torso to get minimal body size. Picked this by experimentation.
  private static final float TORSO_MULTIPLIER = 2.5f;

  public static List<PointF3D> getPoseEmbedding(List<PointF3D> landmarks) {
    List<PointF3D> normalizedLandmarks = normalize(landmarks);
    return getEmbedding(normalizedLandmarks);
  }

  private static List<PointF3D> normalize(List<PointF3D> landmarks) {
    List<PointF3D> normalizedLandmarks = new ArrayList<>(landmarks);
    // Normalize translation.
    PointF3D center = average(
        landmarks.get(PoseLandmark.LEFT_HIP), landmarks.get(PoseLandmark.RIGHT_HIP));
    subtractAll(center, normalizedLandmarks);

    // Normalize scale.
    multiplyAll(normalizedLandmarks, 1 / getPoseSize(normalizedLandmarks));
    // Multiplication by 100 is not required, but makes it easier to debug.
    multiplyAll(normalizedLandmarks, 100);
    return normalizedLandmarks;
  }

  // Translation normalization should've been done prior to calling this method.
  private static float getPoseSize(List<PointF3D> landmarks) {
    // Note: This approach uses only 2D landmarks to compute pose size as using Z wasn't helpful
    // in our experimentation but you're welcome to tweak.
    PointF3D hipsCenter = average(
        landmarks.get(PoseLandmark.LEFT_HIP), landmarks.get(PoseLandmark.RIGHT_HIP));

    PointF3D shouldersCenter = average(
        landmarks.get(PoseLandmark.LEFT_SHOULDER),
        landmarks.get(PoseLandmark.RIGHT_SHOULDER));

    float torsoSize = l2Norm2D(subtract(hipsCenter, shouldersCenter));

    float maxDistance = torsoSize * TORSO_MULTIPLIER;
    // torsoSize * TORSO_MULTIPLIER is the floor we want based on experimentation but actual size
    // can be bigger for a given pose depending on extension of limbs etc so we calculate that.
    for (PointF3D landmark : landmarks) {
      float distance = l2Norm2D(subtract(hipsCenter, landmark));
      if (distance > maxDistance) {
        maxDistance = distance;
      }
    }
    return maxDistance;
  }

  private static List<PointF3D> getEmbedding(List<PointF3D> lm) {
    List<PointF3D> embedding = new ArrayList<>();

    // We use several pairwise 3D distances to form pose embedding. These were selected
    // based on experimentation for best results with our default pose classes as captued in the
    // pose samples csv. Feel free to play with this and add or remove for your use-cases.

    // We group our distances by number of joints between the pairs.
    // One joint.
    embedding.add(subtract(
        average(lm.get(PoseLandmark.LEFT_HIP), lm.get(PoseLandmark.RIGHT_HIP)),
        average(lm.get(PoseLandmark.LEFT_SHOULDER), lm.get(PoseLandmark.RIGHT_SHOULDER))
    ));

    embedding.add(subtract(
        lm.get(PoseLandmark.LEFT_SHOULDER), lm.get(PoseLandmark.LEFT_ELBOW)));
    embedding.add(subtract(
        lm.get(PoseLandmark.RIGHT_SHOULDER), lm.get(PoseLandmark.RIGHT_ELBOW)));

    embedding.add(subtract(lm.get(PoseLandmark.LEFT_ELBOW), lm.get(PoseLandmark.LEFT_WRIST)));
    embedding.add(subtract(lm.get(PoseLandmark.RIGHT_ELBOW), lm.get(PoseLandmark.RIGHT_WRIST)));

    embedding.add(subtract(lm.get(PoseLandmark.LEFT_HIP), lm.get(PoseLandmark.LEFT_KNEE)));
    embedding.add(subtract(lm.get(PoseLandmark.RIGHT_HIP), lm.get(PoseLandmark.RIGHT_KNEE)));

    embedding.add(subtract(lm.get(PoseLandmark.LEFT_KNEE), lm.get(PoseLandmark.LEFT_ANKLE)));
    embedding.add(subtract(lm.get(PoseLandmark.RIGHT_KNEE), lm.get(PoseLandmark.RIGHT_ANKLE)));

    // Two joints.
    embedding.add(subtract(
        lm.get(PoseLandmark.LEFT_SHOULDER), lm.get(PoseLandmark.LEFT_WRIST)));
    embedding.add(subtract(
        lm.get(PoseLandmark.RIGHT_SHOULDER), lm.get(PoseLandmark.RIGHT_WRIST)));

    embedding.add(subtract(lm.get(PoseLandmark.LEFT_HIP), lm.get(PoseLandmark.LEFT_ANKLE)));
    embedding.add(subtract(lm.get(PoseLandmark.RIGHT_HIP), lm.get(PoseLandmark.RIGHT_ANKLE)));

    // Four joints.
    embedding.add(subtract(lm.get(PoseLandmark.LEFT_HIP), lm.get(PoseLandmark.LEFT_WRIST)));
    embedding.add(subtract(lm.get(PoseLandmark.RIGHT_HIP), lm.get(PoseLandmark.RIGHT_WRIST)));

    // Five joints.
    embedding.add(subtract(
        lm.get(PoseLandmark.LEFT_SHOULDER), lm.get(PoseLandmark.LEFT_ANKLE)));
    embedding.add(subtract(
        lm.get(PoseLandmark.RIGHT_SHOULDER), lm.get(PoseLandmark.RIGHT_ANKLE)));

    embedding.add(subtract(lm.get(PoseLandmark.LEFT_HIP), lm.get(PoseLandmark.LEFT_WRIST)));
    embedding.add(subtract(lm.get(PoseLandmark.RIGHT_HIP), lm.get(PoseLandmark.RIGHT_WRIST)));

    // Cross body.
    embedding.add(subtract(lm.get(PoseLandmark.LEFT_ELBOW), lm.get(PoseLandmark.RIGHT_ELBOW)));
    embedding.add(subtract(lm.get(PoseLandmark.LEFT_KNEE), lm.get(PoseLandmark.RIGHT_KNEE)));

    embedding.add(subtract(lm.get(PoseLandmark.LEFT_WRIST), lm.get(PoseLandmark.RIGHT_WRIST)));
    embedding.add(subtract(lm.get(PoseLandmark.LEFT_ANKLE), lm.get(PoseLandmark.RIGHT_ANKLE)));

    return embedding;
  }

  private PoseEmbedding() {}
}


//代码解析：PoseEmbedding 类用于姿势嵌入生成
//这段 Java 代码定义了一个名为 PoseEmbedding 的类，其作用是从一组姿势关键点生成嵌入向量 (embedding)。嵌入向量可以用于表示姿势特征，并用于后续的分类、相似度比较等任务。
//主要步骤：
//标准化 (normalize):
//将所有关键点减去躯干中心的坐标，进行平移标准化。
//计算标准化后的关键点到躯干中心的距离，并将其乘以一个常数因子 (TORSO_MULTIPLIER)，得到一个最小身体尺寸。
//找到所有关键点到躯干中心的距离的最大值，作为标准化尺度。
//将所有关键点坐标除以标准化尺度，进行缩放标准化。
//最后将所有坐标乘以100，方便调试。
//生成嵌入向量 (getEmbedding):
//计算不同关键点之间的距离，例如：
//相邻关键点之间的距离 (例如肩膀和手肘)
//相隔多个关键点之间的距离 (例如肩膀和手腕)
//跨身体的距离 (例如左肘和右肘)
//将这些距离组合成一个向量，即嵌入向量。
//代码解读：
//getPoseEmbedding(landmarks)：这个方法接受一个 PointF3D 类型的列表作为输入，代表所有关键点的坐标，并返回一个同样是 PointF3D 类型的列表，代表生成的嵌入向量。
//normalize(landmarks)：这个方法进行关键点坐标的标准化，包括平移和缩放。
//getPoseSize(landmarks)：这个方法计算标准化后的身体尺寸，用于缩放标准化。
//getEmbedding(lm)：这个方法根据标准化后的关键点计算嵌入向量，包含了各种关键点之间的距离。
//代码目的：
//通过将姿势关键点转换为嵌入向量，可以方便地进行后续的姿势分类、相似度比较等任务。例如，可以将嵌入向量输入到机器学习模型中，进行姿势识别或动作分析。
//代码应用：
//这个代码可以用于各种姿势相关的应用，例如：
//健身应用: 分析用户的健身动作是否标准。
//运动分析: 分析运动员的动作，提供训练建议。
//人机交互: 使用姿势进行控制或交互。
//虚拟现实: 将用户的姿势映射到虚拟角色上。
//总结：
//PoseEmbedding 类提供了一种有效的方法，将姿势关键点转换为嵌入向量，方便进行后续的姿势分析和应用。代码结构清晰，易于理解和修改，可以作为姿势相关应用开发的参考。