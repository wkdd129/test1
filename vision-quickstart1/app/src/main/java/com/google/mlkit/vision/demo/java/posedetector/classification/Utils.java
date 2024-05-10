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

import static com.google.common.primitives.Floats.max;

import com.google.mlkit.vision.common.PointF3D;
import java.util.List;
import java.util.ListIterator;

/**
 * Utility methods for operations on {@link PointF3D}.
 */
public class Utils {
  private Utils() {}

  public static PointF3D add(PointF3D a, PointF3D b) {
    return PointF3D.from(a.getX() + b.getX(), a.getY() + b.getY(), a.getZ() + b.getZ());
  }

  public static PointF3D subtract(PointF3D b, PointF3D a) {
    return PointF3D.from(a.getX() - b.getX(), a.getY() - b.getY(), a.getZ() - b.getZ());
  }

  public static PointF3D multiply(PointF3D a, float multiple) {
    return PointF3D.from(a.getX() * multiple, a.getY() * multiple, a.getZ() * multiple);
  }

  public static PointF3D multiply(PointF3D a, PointF3D multiple) {
    return PointF3D.from(
        a.getX() * multiple.getX(), a.getY() * multiple.getY(), a.getZ() * multiple.getZ());
  }

  public static PointF3D average(PointF3D a, PointF3D b) {
    return PointF3D.from(
        (a.getX() + b.getX()) * 0.5f, (a.getY() + b.getY()) * 0.5f, (a.getZ() + b.getZ()) * 0.5f);
  }

  public static float l2Norm2D(PointF3D point) {
    return (float) Math.hypot(point.getX(), point.getY());
  }

  public static float maxAbs(PointF3D point) {
    return max(Math.abs(point.getX()), Math.abs(point.getY()), Math.abs(point.getZ()));
  }

  public static float sumAbs(PointF3D point) {
    return Math.abs(point.getX()) + Math.abs(point.getY()) + Math.abs(point.getZ());
  }

  public static void addAll(List<PointF3D> pointsList, PointF3D p) {
    ListIterator<PointF3D> iterator = pointsList.listIterator();
    while (iterator.hasNext()) {
      iterator.set(add(iterator.next(), p));
    }
  }

  public static void subtractAll(PointF3D p, List<PointF3D> pointsList) {
    ListIterator<PointF3D> iterator = pointsList.listIterator();
    while (iterator.hasNext()) {
      iterator.set(subtract(p, iterator.next()));
    }
  }

  public static void multiplyAll(List<PointF3D> pointsList, float multiple) {
    ListIterator<PointF3D> iterator = pointsList.listIterator();
    while (iterator.hasNext()) {
      iterator.set(multiply(iterator.next(), multiple));
    }
  }

  public static void multiplyAll(List<PointF3D> pointsList, PointF3D multiple) {
    ListIterator<PointF3D> iterator = pointsList.listIterator();
    while (iterator.hasNext()) {
      iterator.set(multiply(iterator.next(), multiple));
    }
  }
}
//
//代码解析：Utils 类提供 PointF3D 类型的工具类
//这段 Java 代码定义了一个名为 Utils 的类，它提供了一系列静态方法，用于操作 PointF3D 类型的数据。PointF3D 类型表示三维空间中的一个点，具有 x、y 和 z 三个坐标值。
//主要功能：
//基本运算：
//add(a, b)：将两个 PointF3D 对象相加。
//subtract(b, a)：从 a 中减去 b。
//multiply(a, multiple)：将 a 乘以一个标量。
//multiply(a, multiple)：将 a 与另一个 PointF3D 对象逐元素相乘。
//average(a, b)：计算两个 PointF3D 对象的平均值。
//距离计算：
//l2Norm2D(point)：计算二维平面上点到原点的欧几里得距离。
//取值操作：
//maxAbs(point)：返回点坐标绝对值的最大值。
//sumAbs(point)：返回点坐标绝对值的总和。
//列表操作：
//addAll(pointsList, p)：将 p 加到 pointsList 中的每个点上。
//subtractAll(p, pointsList)：从 pointsList 中的每个点减去 p。
//multiplyAll(pointsList, multiple)：将 pointsList 中的每个点乘以一个标量。
//multiplyAll(pointsList, multiple)：将 pointsList 中的每个点与另一个 PointF3D 对象逐元素相乘。
//代码应用：
//Utils 类中的方法可以用于各种需要处理三维点数据的场景，例如：
//姿势识别: 计算人体关键点之间的距离，或对关键点坐标进行归一化处理。
//三维图形: 进行点运算、向量运算和矩阵运算。
//游戏开发: 处理游戏对象的位置和运动。
//计算机视觉: 进行图像处理和三维重建。
//总结：
//Utils 类提供了一组方便的工具方法，用于操作 PointF3D 类型的数据。这些方法可以简化三维点数据的处理过程，并提高代码的可读性和可维护性。