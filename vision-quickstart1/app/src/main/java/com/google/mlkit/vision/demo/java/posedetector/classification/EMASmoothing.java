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

import android.os.SystemClock;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * Runs EMA smoothing over a window with given stream of pose classification results.
 */
public class EMASmoothing {
  private static final int DEFAULT_WINDOW_SIZE = 10;
  private static final float DEFAULT_ALPHA = 0.2f;

  private static final long RESET_THRESHOLD_MS = 100;

  private final int windowSize;
  private final float alpha;
  // This is a window of {@link ClassificationResult}s as outputted by the {@link PoseClassifier}.
  // We run smoothing over this window of size {@link windowSize}.
  private final Deque<ClassificationResult> window;

  private long lastInputMs;

  public EMASmoothing() {
    this(DEFAULT_WINDOW_SIZE, DEFAULT_ALPHA);
  }

  public EMASmoothing(int windowSize, float alpha) {
    this.windowSize = windowSize;
    this.alpha = alpha;
    this.window = new LinkedBlockingDeque<>(windowSize);
  }

  public ClassificationResult getSmoothedResult(ClassificationResult classificationResult) {
    // Resets memory if the input is too far away from the previous one in time.
    long nowMs = SystemClock.elapsedRealtime();
    if (nowMs - lastInputMs > RESET_THRESHOLD_MS) {
      window.clear();
    }
    lastInputMs = nowMs;

    // If we are at window size, remove the last (oldest) result.
    if (window.size() == windowSize) {
      window.pollLast();
    }
    // Insert at the beginning of the window.
    window.addFirst(classificationResult);

    Set<String> allClasses = new HashSet<>();
    for (ClassificationResult result : window) {
      allClasses.addAll(result.getAllClasses());
    }

    ClassificationResult smoothedResult = new ClassificationResult();

    for (String className : allClasses) {
      float factor = 1;
      float topSum = 0;
      float bottomSum = 0;
      for (ClassificationResult result : window) {
        float value = result.getClassConfidence(className);

        topSum += factor * value;
        bottomSum += factor;

        factor = (float) (factor * (1.0 - alpha));
      }
      smoothedResult.putClassConfidence(className, topSum / bottomSum);
    }

    return smoothedResult;
  }
}
//代码解析：EMASmoothing 类用于对姿势分类结果进行指数移动平均平滑
//这段 Java 代码定义了一个名为 EMASmoothing 的类，其作用是对一系列姿势分类结果进行指数移动平均 (EMA) 平滑处理。EMA 是一种常用的时间序列数据平滑方法，可
// 以减少噪声的影响，使结果更加稳定。
//主要功能：
//维护一个结果窗口： window 存储了最近的 windowSize 个 ClassificationResult 对象。
//执行 EMA 平滑： getSmoothedResult(classificationResult) 方法接受一个新的分类结果，并将其加入窗口。然后，它计算所有类别的 EMA 平滑值
// ，并返回一个新的 ClassificationResult 对象，其中包含平滑后的置信度。
//重置机制： 如果当前输入与上一次输入的时间间隔超过 RESET_THRESHOLD_MS，则清空窗口，重新开始平滑。
//代码解读：
//windowSize：窗口大小，即存储的分类结果数量。
//alpha：EMA 平滑因子，控制平滑程度。较大的 alpha 值会使平滑结果更接近于最新数据，而较小的 alpha 值会使平滑结果更加平稳。
//window：一个双端队列，用于存储最近的分类结果。
//lastInputMs：上一次输入的时间戳。
//getSmoothedResult(classificationResult)：
//检查当前输入与上一次输入的时间间隔，如果超过阈值，则清空窗口。
//将新的分类结果加入窗口，如果窗口已满，则移除最旧的结果。
//遍历所有类别，计算每个类别的 EMA 平滑值：
//factor：权重因子，随着时间的推移呈指数衰减。
//topSum：加权置信度之和。
//bottomSum：权重因子之和。
//将平滑后的置信度存储到新的 ClassificationResult 分类结果 对象中，并返回。
//代码应用：
//这个代码可以用于各种需要平滑处理姿势分类结果的应用，例如：
//健身应用: 减少姿势分类结果的抖动，使动作识别更稳定。
//运动分析: 平滑运动员的动作数据，便于观察趋势和变化。
//人机交互: 提高姿势控制的稳定性。
//虚拟现实: 使虚拟角色的动作更流畅。
//总结：
//EMASmoothing EMA平滑 类提供了一种简单有效的方法，对姿势分类结果进行平滑处理，可以提高结果的稳定性和可靠性。它可以作为姿势识别和分析应用中的一个重要组件。
