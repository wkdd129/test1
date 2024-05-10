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

import static com.google.mlkit.vision.demo.java.posedetector.classification.PoseEmbedding.getPoseEmbedding;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.maxAbs;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.multiply;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.multiplyAll;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.subtract;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.sumAbs;
import static java.lang.Math.max;
import static java.lang.Math.min;

import android.util.Pair;
import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseLandmark;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Classifies {link Pose} based on given {@link PoseSample}s.
 *
 * <p>Inspired by K-Nearest Neighbors Algorithm with outlier filtering.
 * https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
 */
public class PoseClassifier {
  private static final String TAG = "PoseClassifier";
  private static final int MAX_DISTANCE_TOP_K = 30;
  private static final int MEAN_DISTANCE_TOP_K = 10;
  // Note Z has a lower weight as it is generally less accurate than X & Y.
  private static final PointF3D AXES_WEIGHTS = PointF3D.from(1, 1, 0.2f);

  private final List<PoseSample> poseSamples;
  private final int maxDistanceTopK;
  private final int meanDistanceTopK;
  private final PointF3D axesWeights;

  public PoseClassifier(List<PoseSample> poseSamples) {
    this(poseSamples, MAX_DISTANCE_TOP_K, MEAN_DISTANCE_TOP_K, AXES_WEIGHTS);
  }

  public PoseClassifier(List<PoseSample> poseSamples, int maxDistanceTopK,
      int meanDistanceTopK, PointF3D axesWeights) {
    this.poseSamples = poseSamples;
    this.maxDistanceTopK = maxDistanceTopK;
    this.meanDistanceTopK = meanDistanceTopK;
    this.axesWeights = axesWeights;
  }

  private static List<PointF3D> extractPoseLandmarks(Pose pose) {
    List<PointF3D> landmarks = new ArrayList<>();
    for (PoseLandmark poseLandmark : pose.getAllPoseLandmarks()) {
      landmarks.add(poseLandmark.getPosition3D());
    }
    return landmarks;
  }

  /**
   * Returns the max range of confidence values.
   *
   * <p><Since we calculate confidence by counting {@link PoseSample}s that survived
   * outlier-filtering by maxDistanceTopK and meanDistanceTopK, this range is the minimum of two.
   */
  public int confidenceRange() {
    return min(maxDistanceTopK, meanDistanceTopK);
  }

  public ClassificationResult classify(Pose pose) {
    return classify(extractPoseLandmarks(pose));
  }

  public ClassificationResult classify(List<PointF3D> landmarks) {
    ClassificationResult result = new ClassificationResult();
    // Return early if no landmarks detected.
    if (landmarks.isEmpty()) {
      return result;
    }

    // We do flipping on X-axis so we are horizontal (mirror) invariant.
    List<PointF3D> flippedLandmarks = new ArrayList<>(landmarks);
    multiplyAll(flippedLandmarks, PointF3D.from(-1, 1, 1));

    List<PointF3D> embedding = getPoseEmbedding(landmarks);
    List<PointF3D> flippedEmbedding = getPoseEmbedding(flippedLandmarks);


    // Classification is done in two stages:
    //  * First we pick top-K samples by MAX distance. It allows to remove samples that are almost
    //    the same as given pose, but maybe has few joints bent in the other direction.
    //  * Then we pick top-K samples by MEAN distance. After outliers are removed, we pick samples
    //    that are closest by average.

    // Keeps max distance on top so we can pop it when top_k size is reached.
    PriorityQueue<Pair<PoseSample, Float>> maxDistances = new PriorityQueue<>(
        maxDistanceTopK, (o1, o2) -> -Float.compare(o1.second, o2.second));
    // Retrieve top K poseSamples by least distance to remove outliers.
    for (PoseSample poseSample : poseSamples) {
      List<PointF3D> sampleEmbedding = poseSample.getEmbedding();

      float originalMax = 0;
      float flippedMax = 0;
      for (int i = 0; i < embedding.size(); i++) {
        originalMax =
            max(
                originalMax,
                maxAbs(multiply(subtract(embedding.get(i), sampleEmbedding.get(i)), axesWeights)));
        flippedMax =
            max(
                flippedMax,
                maxAbs(
                    multiply(
                        subtract(flippedEmbedding.get(i), sampleEmbedding.get(i)), axesWeights)));
      }
      // Set the max distance as min of original and flipped max distance.
      maxDistances.add(new Pair<>(poseSample, min(originalMax, flippedMax)));
      // We only want to retain top n so pop the highest distance.
      if (maxDistances.size() > maxDistanceTopK) {
        maxDistances.poll();
      }
    }

    // Keeps higher mean distances on top so we can pop it when top_k size is reached.
    PriorityQueue<Pair<PoseSample, Float>> meanDistances = new PriorityQueue<>(
        meanDistanceTopK, (o1, o2) -> -Float.compare(o1.second, o2.second));
    // Retrive top K poseSamples by least mean distance to remove outliers.
    for (Pair<PoseSample, Float> sampleDistances : maxDistances) {
      PoseSample poseSample = sampleDistances.first;
      List<PointF3D> sampleEmbedding = poseSample.getEmbedding();

      float originalSum = 0;
      float flippedSum = 0;
      for (int i = 0; i < embedding.size(); i++) {
        originalSum += sumAbs(multiply(
            subtract(embedding.get(i), sampleEmbedding.get(i)), axesWeights));
        flippedSum += sumAbs(
            multiply(subtract(flippedEmbedding.get(i), sampleEmbedding.get(i)), axesWeights));
      }
      // Set the mean distance as min of original and flipped mean distances.
      float meanDistance = min(originalSum, flippedSum) / (embedding.size() * 2);
      meanDistances.add(new Pair<>(poseSample, meanDistance));
      // We only want to retain top k so pop the highest mean distance.
      if (meanDistances.size() > meanDistanceTopK) {
        meanDistances.poll();
      }
    }

    for (Pair<PoseSample, Float> sampleDistances : meanDistances) {
      String className = sampleDistances.first.getClassName();
      result.incrementClassConfidence(className);
    }

    return result;
  }
}
//代码解析：PoseClassifier 类用于姿势分类
//这段 Java 代码定义了一个名为 PoseClassifier 的类，其作用是根据给定的姿势样本 (PoseSample) 对输入的姿势 (Pose) 进行
// 分类。它采用了 K-Nearest Neighbors (KNN) 算法的思想，并结合了异常值过滤，以提高分类的准确性。
//主要步骤：
//提取姿势关键点： extractPoseLandmarks(pose) 方法从 Pose 对象中提取所有关键点的三维坐标，存储在一个 PointF3D 类型的列表中。
//生成姿势嵌入： 使用 PoseEmbedding.getPoseEmbedding(landmarks) 方法将关键点坐标转换为嵌入向量，用于表示姿势特征。
//镜像处理： 将关键点坐标沿 X 轴镜像翻转，生成一个镜像的嵌入向量。这是为了使分类结果不受人体朝向的影响。
//KNN 分类：
//最大距离过滤：
//使用优先队列 maxDistances 存储距离输入姿势最近的 maxDistanceTopK 个样本及其距离。
//对于每个样本，计算其嵌入向量与输入姿势嵌入向量以及镜像嵌入向量之间的最大距离。
//取两者中的较小值作为该样本与输入姿势的距离。
//如果队列大小超过 maxDistanceTopK，则移除距离最远的样本。
//平均距离过滤：
//使用优先队列 meanDistances 存储平均距离输入姿势最近的 meanDistanceTopK 个样本及其距离。
//对于每个样本，计算其嵌入向量与输入姿势嵌入向量以及镜像嵌入向量之间的平均距离。
//取两者中的较小值作为该样本与输入姿势的距离。
//如果队列大小超过 meanDistanceTopK，则移除平均距离最远的样本。
//置信度计算：
//遍历 meanDistances 队列中的所有样本，将其所属的类别置信度加 1。
//返回分类结果： 将包含所有类别及其置信度的 ClassificationResult 对象返回。
//代码解读：
//poseSamples：存储所有姿势样本的列表。
//maxDistanceTopK：最大距离过滤中选择的样本数量。
//meanDistanceTopK：平均距离过滤中选择的样本数量。
//axesWeights：用于加权不同坐标轴的权重向量。
//confidenceRange()：返回置信度值的范围，即 min(maxDistanceTopK, meanDistanceTopK)。
//classify(pose)：根据输入的 Pose 对象进行分类。
//classify(landmarks)：根据输入的关键点坐标进行分类。
//代码应用：
//这个代码可以用于各种需要姿势分类的应用，例如：
//健身应用: 识别用户的健身动作，并判断动作是否标准。
//运动分析: 分析运动员的动作，提供训练建议。
//人机交互: 使用姿势进行控制或交互。
//虚拟现实: 将用户的姿势映射到虚拟角色上。
//总结：
//PoseClassifier 类提供了一种基于 KNN 算法和异常值过滤的姿势分类方法。它能够有效地识别不同类型的姿势，并提供可靠的置信度估计，可以作为姿势识别和分析应用中的核心组件。