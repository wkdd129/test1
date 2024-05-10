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

/**
 * Counts reps for the give class.
 */
public class RepetitionCounter {
  // These thresholds can be tuned in conjunction with the Top K values in {@link PoseClassifier}.
  // The default Top K value is 10 so the range here is [0-10].
  private static final float DEFAULT_ENTER_THRESHOLD = 6f;
  private static final float DEFAULT_EXIT_THRESHOLD = 4f;

  private final String className;
  private final float enterThreshold;
  private final float exitThreshold;

  private int numRepeats;
  private boolean poseEntered;

  public RepetitionCounter(String className) {
    this(className, DEFAULT_ENTER_THRESHOLD, DEFAULT_EXIT_THRESHOLD);
  }

  public RepetitionCounter(String className, float enterThreshold, float exitThreshold) {
    this.className = className;
    this.enterThreshold = enterThreshold;
    this.exitThreshold = exitThreshold;
    numRepeats = 0;
    poseEntered = false;
  }

  /**
   * Adds a new Pose classification result and updates reps for given class.
   *
   * @param classificationResult {link ClassificationResult} of class to confidence values.
   * @return number of reps.
   */
  public int addClassificationResult(ClassificationResult classificationResult) {
    float poseConfidence = classificationResult.getClassConfidence(className);

    if (!poseEntered) {
      poseEntered = poseConfidence > enterThreshold;
      return numRepeats;
    }

    if (poseConfidence < exitThreshold) {
      numRepeats++;
      poseEntered = false;
    }

    return numRepeats;
  }

  public String getClassName() {
    return className;
  }

  public int getNumRepeats() {
    return numRepeats;
  }
}
//代码解析：RepetitionCounter 类用于计算重复动作次数
//这段 Java 代码定义了一个名为 RepetitionCounter 的类，其作用是针对给定的姿势类别 (例如俯卧撑或深蹲)，计算重复动作的次数。它通过跟踪姿势的进入和退出状态，以及置信度阈值，来判断是否完成了一个重复动作。
//主要功能：
//初始化： 接受姿势类别名称、进入阈值和退出阈值作为参数。
//添加分类结果： addClassificationResult(classificationResult) 方法接受一个 ClassificationResult 对象，并根据置信度更新重复计数。
//获取信息： 提供 getClassName() 和 getNumRepeats() 方法，分别用于获取姿势类别名称和重复次数。
//代码解读：
//className：姿势类别名称，例如 "pushups" 或 "squats"。
//enterThreshold：进入阈值，当姿势置信度超过该阈值时，认为进入了该姿势。
//exitThreshold：退出阈值，当姿势置信度低于该阈值时，认为退出了该姿势。
//numRepeats：重复次数。
//poseEntered：是否已进入该姿势。
//addClassificationResult(classificationResult)
//添加分类结果（分类结果）：
//获取指定姿势类别的置信度。
//如果尚未进入该姿势，且置信度超过进入阈值，则将 poseEntered 姿势已输入 设置为 true。
//如果已进入该姿势，且置信度低于退出阈值，则将 numRepeats 加 1，并将 poseEntered 设置为 false，表示完成了一个重复动作。
//返回当前的重复次数。
//代码应用：
//RepetitionCounter 类可以用于各种需要进行重复动作计数的应用，例如：
//健身应用: 跟踪用户的健身进度，例如计算完成的俯卧撑或深蹲次数。
//运动分析: 分析运动员的动作，例如计算举重的次数。
//康复训练: 帮助患者进行康复训练，并监测训练量。
//总结：
//RepetitionCounter 类提供了一种简单有效的方法，用于计算给定姿势类别的重复动作次数。
// 它通过置信度阈值和状态机来判断是否完成了一个重复动作，可以方便地集成到各种姿势识别和分析应用中。

