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

import android.util.Log;
import com.google.common.base.Splitter;
import com.google.mlkit.vision.common.PointF3D;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads Pose samples from a csv file.
 */
public class PoseSample {
  private static final String TAG = "PoseSample";
  private static final int NUM_LANDMARKS = 33;
  private static final int NUM_DIMS = 3;

  private final String name;
  private final String className;
  private final List<PointF3D> embedding;

  public PoseSample(String name, String className, List<PointF3D> landmarks) {
    this.name = name;
    this.className = className;
    this.embedding = PoseEmbedding.getPoseEmbedding(landmarks);
  }

  public String getName() {
    return name;
  }

  public String getClassName() {
    return className;
  }

  public List<PointF3D> getEmbedding() {
    return embedding;
  }

  public static PoseSample getPoseSample(String csvLine, String separator) {
    List<String> tokens = Splitter.onPattern(separator).splitToList(csvLine);
    // Format is expected to be Name,Class,X1,Y1,Z1,X2,Y2,Z2...
    // + 2 is for Name & Class.
    if (tokens.size() != (NUM_LANDMARKS * NUM_DIMS) + 2) {
      Log.e(TAG, "Invalid number of tokens for PoseSample");
      return null;
    }
    String name = tokens.get(0);
    String className = tokens.get(1);
    List<PointF3D> landmarks = new ArrayList<>();
    // Read from the third token, first 2 tokens are name and class.
    for (int i = 2; i < tokens.size(); i += NUM_DIMS) {
      try {
        landmarks.add(
            PointF3D.from(
                Float.parseFloat(tokens.get(i)),
                Float.parseFloat(tokens.get(i + 1)),
                Float.parseFloat(tokens.get(i + 2))));
      } catch (NullPointerException | NumberFormatException e) {
        Log.e(TAG, "Invalid value " + tokens.get(i) + " for landmark position.");
        return null;
      }
    }
    return new PoseSample(name, className, landmarks);
  }
}
//
//代码解析: PoseSample 类用于表示姿势样本数据
//这段 Java 代码定义了一个名为 PoseSample 的类，其作用是表示一个姿势样本数据。每个样本包含以下信息：
//名称 (name): 样本的名称，例如 "pushup_down" 或 "squat_down"。
//类别 (className): 样本所属的类别，例如 "pushups" 或 "squats"。
//嵌入向量 (embedding): 表示样本特征的嵌入向量，由 PoseEmbedding 姿势嵌入 类生成。
//主要功能：
//构造函数: 接受样本名称、类别和关键点坐标作为参数，并使用 PoseEmbedding 类生成嵌入向量。
//获取方法: 提供 getName()、getClassName() 和 getEmbedding() 方法，分别用于获取样本的名称、类别和嵌入向量。
//静态方法 getPoseSample(csvLine, separator): 从 csv 文件的一行数据中解析出姿势样本信息，并创建一个 PoseSample 对象。
//代码解读：
//NUM_LANDMARKS：人体关键点数量，此处为 33。
//NUM_DIMS：每个关键点的维度，此处为 3 (x, y, z)。
//name：样本名称。
//className：样本类别。
//embedding：样本的嵌入向量。
//getPoseSample(csvLine, separator)：
//使用 Splitter 类将 csv 数据行分割成多个字符串。
//检查字符串数量是否正确，即关键点坐标数量加上名称和类别。
//解析出样本名称和类别。
//遍历剩余的字符串，解析出每个关键点的坐标，并创建 PointF3D 对象。
//创建一个 PoseSample 对象，并返回。
//代码应用：
//PoseSample 类是 PoseClassifier 的基础，用于存储和表示姿势样本数据。它可以用于以下场景：
//训练姿势分类器: 将多个姿势样本数据存储为 PoseSample 对象，并用于训练 PoseClassifier。
//评估姿势分类器: 使用 PoseSample 对象测试 PoseClassifier 的准确性。
//可视化姿势样本: 将 PoseSample 对象的关键点坐标可视化，以便观察样本的姿势。
//总结：
//PoseSample 类提供了一种方便的方式来表示姿势样本数据，可以用于训练和评估姿势分类器，以及可视化姿势样本。它是构建姿势识别和分析应用的重要组件。
