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

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static java.util.Collections.max;

/**
 * Represents Pose classification result as outputted by {@link PoseClassifier}. Can be manipulated.
 */
public class ClassificationResult {
  // For an entry in this map, the key is the class name, and the value is how many times this class
  // appears in the top K nearest neighbors. The value is in range [0, K] and could be a float after
  // EMA smoothing. We use this number to represent the confidence of a pose being in this class.
  private final Map<String, Float> classConfidences;

  public ClassificationResult() {
    classConfidences = new HashMap<>();
  }

  public Set<String> getAllClasses() {
    return classConfidences.keySet();
  }

  public float getClassConfidence(String className) {
    return classConfidences.containsKey(className) ? classConfidences.get(className) : 0;
  }

  public String getMaxConfidenceClass() {
    return max(
        classConfidences.entrySet(),
        (entry1, entry2) -> (int) (entry1.getValue() - entry2.getValue()))
        .getKey();
  }

  public void incrementClassConfidence(String className) {
    classConfidences.put(className,
        classConfidences.containsKey(className) ? classConfidences.get(className) + 1 : 1);
  }

  public void putClassConfidence(String className, float confidence) {
    classConfidences.put(className, confidence);
  }
}

//代码解析: ClassificationResult 类用于存储和操作姿势分类结果
//这个 Java 代码定义了一个名为 ClassificationResult 的类，其作用是存储和操作姿势分类的结果。它主要用于表示 PoseClassifier 类输出的分类结果，并提供一些方
// 法来获取和修改分类结果的置信度。
//主要功能：
//存储分类结果: classConfidences 字典用于存储每个姿势类别的置信度。字典的键是类别的名称 (String 类型)，值是该类别出现的次数 (Float 类型)。
//获取所有类别: getAllClasses() 方法返回所有已识别姿势类别的集合 (Set 类型)。
//获取特定类别的置信度: getClassConfidence(className) 方法返回指定类别的置信度 (Float 类型)。如果没有该类别，则返回 0。
//获取置信度最高的类别: getMaxConfidenceClass() 方法返回置信度最高的类别的名称 (String 类型)。
//增加类别的置信度: incrementClassConfidence(className) 方法将指定类别的置信度加 1。
//设置类别的置信度: putClassConfidence(className, confidence) 方法将指定类别的置信度设置为给定的值。
//代码解读：
//classConfidences：这个字典用于存储每个类别的置信度。置信度可以是整数，表示该类别在K个最近邻中出现的次数；也可以是浮点数，表示经过指数移动平均 (EMA) 平滑后的置信度。
//getAllClasses()：这个方法返回 classConfidences 字典的所有键，即所有已识别的类别。
//getClassConfidence(className)：这个方法返回指定类别的置信度。如果 classConfidences 字典中不存在该类别，则返回 0。
//getMaxConfidenceClass()：这个方法使用 lambda 表达式找到 classConfidences 字典中值最大的键，即置信度最高的类别。
//incrementClassConfidence(className)：这个方法将指定类别的置信度加 1。如果该类别不存在，则将其置信度设置为 1。
//putClassConfidence(className, confidence)：这个方法将指定类别的置信度设置为给定的值。
//代码应用：
//这个代码可以用于各种姿势分类的应用，例如：
//健身应用: 识别用户的健身动作，并给出反馈。
//运动分析: 分析运动员的动作，判断动作是否标准。
//人机交互: 使用姿势进行控制或交互。
//虚拟现实: 将用户的姿势映射到虚拟角色上。
//总结：
//ClassificationResult 类提供了一种方便的方式来存储和操作姿势分类的结果。它提供了各种方法来获取和修改分类结果的置信度，可以方便地用于各种姿势分类的应用。