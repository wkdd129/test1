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

import android.content.Context;
import android.media.AudioManager;
import android.media.MediaRecorder;
import android.media.ToneGenerator;
import android.os.Looper;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.WorkerThread;
import com.google.common.base.Preconditions;
import com.google.mlkit.vision.pose.Pose;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Accepts a stream of {@link Pose} for classification and Rep counting.
 */
public class PoseClassifierProcessor {
  private static final String TAG = "PoseClassifierProcessor";
  private MediaRecorder mediaRecorder;
  private boolean isRecording = false;
  private static final String POSE_SAMPLES_FILE = "pose/t2.csv";

  // Specify classes for which we want rep counting.
  // These are the labels in the given {@code POSE_SAMPLES_FILE}. You can set your own class labels
  // for your pose samples.
  private static final String PUSHUPS_CLASS = "pushups_down";
  private static final String SQUATS_CLASS = "squats_down";
  private enum PoseState {
    DOWN,UP,T1,T2
  }
  private PoseState currentState = PoseState.DOWN; // 初始状态为 DOWN
  private int repCount = 0; // 动作重复次数
  private static final String[] POSE_CLASSES = {"down", "up", "t2", "t3"};
  private ArrayList<Double> numbers = new ArrayList<>();
  private ArrayList<Double> number2 = new ArrayList<>();
  private ArrayList<Double> number3 = new ArrayList<>();
  private ArrayList<Double> number4 = new ArrayList<>();
  private float revie = 0; // 动作重复次数
  private Context context;
  private final boolean isStreamMode;

  private EMASmoothing emaSmoothing;
  private List<RepetitionCounter> repCounters;
  private PoseClassifier poseClassifier;
  private String lastRepResult;



  private void configureMediaRecorder() {
    // ... existing code (setAudioSource, setVideoSource, etc.) ...
    mediaRecorder.setOutputFile(getOutputMediaFile().toString());
    // ... existing code (setVideoEncodingBitRate, setVideoFrameRate, etc.) ...

    try {
      mediaRecorder.prepare();
    } catch (IOException e) {
      Log.e(TAG, "MediaRecorder prepare failed:", e);
    }
  }

  private File getOutputMediaFile() {
    // 使用 Context.getExternalFilesDir() 获取应用可访问的外部存储路径
    File mediaStorageDir = new File(context.getExternalFilesDir(null), "test");
    if (!mediaStorageDir.exists()) {
      if (!mediaStorageDir.mkdirs()) {
        Log.d(TAG, "failed to create directory");
        return null;
      }
    }
    // 使用时间戳创建文件名
    String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
    return new File(mediaStorageDir.getPath() + File.separator +
            "VID_" + timeStamp + ".mp4");
  }

  private List<Double> actionDurations = new ArrayList<>(); // 记录每个动作的持续时间

  private double startTime = 0;
  private double lastActionTime = 0;
  @WorkerThread
  public PoseClassifierProcessor(Context context, boolean isStreamMode) {
    this.context = context;
    Preconditions.checkState(Looper.myLooper() != Looper.getMainLooper());
    this.isStreamMode = isStreamMode;
    if (isStreamMode) {
      emaSmoothing = new EMASmoothing();
      repCounters = new ArrayList<>();
      lastRepResult = "";
    }
    loadPoseSamples(context);
  }

  private void loadPoseSamples(Context context) {
    List<PoseSample> poseSamples = new ArrayList<>();
    try {
      BufferedReader reader = new BufferedReader(
              new InputStreamReader(context.getAssets().open(POSE_SAMPLES_FILE)));
      String csvLine = reader.readLine();
      while (csvLine != null) {
        // If line is not a valid {@link PoseSample}, we'll get null and skip adding to the list.
        PoseSample poseSample = PoseSample.getPoseSample(csvLine, ",");
        if (poseSample != null) {
          poseSamples.add(poseSample);
        }
        csvLine = reader.readLine();
      }
    } catch (IOException e) {
      Log.e(TAG, "Error when loading pose samples.\n" + e);
    }
    poseClassifier = new PoseClassifier(poseSamples);
    if (isStreamMode) {
      for (String className : POSE_CLASSES) {
        repCounters.add(new RepetitionCounter(className));
      }

    }
  }

  /**
   * Given a new {@link Pose} input, returns a list of formatted {@link String}s with Pose
   * classification results.
   *
   * <p>Currently it returns up to 2 strings as following:
   * 0: PoseClass : X reps
   * 1: PoseClass : [0.0-1.0] confidence
   */

  @WorkerThread
  public List<String> getPoseResult(Pose pose) {
    Preconditions.checkState(Looper.myLooper() != Looper.getMainLooper());
    List<String> result = new ArrayList<>();
    ClassificationResult classification = poseClassifier.classify(pose);

    // Update {@link RepetitionCounter}s if {@code isStreamMode}.
    if (isStreamMode) {
      // Feed pose to smoothing even if no pose found.
      classification = emaSmoothing.getSmoothedResult(classification);

      // Return early without updating repCounter if no pose found.
      mediaRecorder = new MediaRecorder();
      for (RepetitionCounter repCounter : repCounters) {
        int repsBefore = repCounter.getNumRepeats();
        int repsAfter = repCounter.addClassificationResult(classification);
        if (repsAfter > repsBefore) {
          // Play a fun beep when rep counter updates.
          ToneGenerator tg = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 100);
          tg.startTone(ToneGenerator.TONE_PROP_BEEP);
          lastRepResult = String.format(
                  Locale.US, "%s : %d reps", repCounter.getClassName(), repsAfter);
          break;
        }
      }
      result.add(lastRepResult);
    }

    // Add maxConfidence class of current frame to result if pose is found.
    if (!pose.getAllPoseLandmarks().isEmpty()) {
      String predictedClass = classification.getMaxConfidenceClass();

      String maxConfidenceClass = classification.getMaxConfidenceClass();

      switch (currentState) {
          case DOWN:
            if (predictedClass.equals("up")) {
              currentState = PoseState.UP;
              if (startTime == 0) {
                startTime = System.currentTimeMillis() / 1000.0;



              }
        // 判断数组内容是否超过10个
              // 判断数组内容是否超过10个
              if (numbers.size() < 10) {
                // 如果没有超过10个，增加1.20这样的数据;
                numbers.add(
                        (double) (classification.getClassConfidence(maxConfidenceClass)
                                                        / poseClassifier.confidenceRange())
                );

              }
            } else {
//              number2 =new ArrayList<>();
//              numbers =new ArrayList<>();
//              number3 =new ArrayList<>();
            }
            break;
          case UP:
            if (predictedClass.equals("t3")) {
              repCount++;
              currentState = PoseState.T1;
//              repCount++;  // 完成一个动作序列，计数加一
              if (number2.size() < 10) {
                // 如果没有超过10个，增加1.20这样的数据
                number2.add(
                        (double) (classification.getClassConfidence(maxConfidenceClass)
                                / poseClassifier.confidenceRange())
                );


                // 计算动作持续时间 (以秒为单位)
                double duration = (System.currentTimeMillis() / 1000.0) - startTime;
                actionDurations.add(duration);
                lastActionTime = System.currentTimeMillis() / 1000.0; // 更新上一个动作的结束时间


              }


            } else {
//              startTime = 0; // 重置起始时间
//              actionDurations.clear(); // 清空动作持续时间
              currentState = PoseState.DOWN; // 重置状态
//              number2 =new ArrayList<>();
//              numbers =new ArrayList<>();
//              number3 =new ArrayList<>();
            }
            break;
          case T1:
            if (predictedClass.equals("t2")) {

              repCount++;  // 完成一个动作序列，计数加一

              currentState = PoseState.T2;
              repCount++;  // 完成一个动作序列，计数加一
              double duration = (System.currentTimeMillis() / 1000.0) - startTime;
              actionDurations.add(duration);
              lastActionTime = System.currentTimeMillis() / 1000.0; // 更新上一个动作的结束时间


            } else {
//              startTime = 0; // 重置起始时间
//              actionDurations.clear(); // 清空动作持续时间
              currentState = PoseState.DOWN; // 重置状态
              float sum = 0;
              for (int i = 0; i < numbers.size(); i++) {
                sum += numbers.get(i);
              }
              float sum1 = 0;
              for (int i = 0; i < number2.size(); i++) {
                sum1 += numbers.get(i);
              }
              revie = (sum / numbers.size() + sum1  / number2.size())/2;

            }
            break;
          case T2:
            if (predictedClass.equals("down")) {
              currentState = PoseState.DOWN;
              repCount++;  // 完成一个动作序列，计数加一
              // 播放提示音或进行其他操作
            } else if (!predictedClass.equals("t1")) {
              // 如果不是 t2 或 down，则重置状态
              currentState = PoseState.DOWN;

            }
          break;

      }



      String maxConfidenceClassResult = String.format(
              Locale.US,
              "%s : %.2f confidence",
              maxConfidenceClass,
              classification.getClassConfidence(maxConfidenceClass)
                      / poseClassifier.confidenceRange()
              );
      result.add(maxConfidenceClassResult);
      result.add("数量是"+repCount);
      result.add("评分为"+revie);
      if (!actionDurations.isEmpty()) {
        for (int i = 0; i < actionDurations.size(); i++) {
          double duration = actionDurations.get(i);
          result.add(String.format(Locale.US, "动作 %d 持续时间：%.2f 秒", i+1, duration));
        }
      }
    }

    return result;
  }



  public interface RepCountListener {
    void onRepCountUpdated(int repCount);
  }
}
//代码解析：PoseClassifierProcessor 类用于处理姿势分类和计数
//这段 Java 代码定义了一个名为 PoseClassifierProcessor 的类，其作用是处理输入的姿势数据，
// 进行分类并计算重复动作次数（例如俯卧撑或深蹲的次数）。它结合了 PoseClassifier 和 RepetitionCounter 等类，实现了对连续姿势流的分析和计数。
//主要功能：
//加载姿势样本： 从指定的文件 (POSE_SAMPLES_FILE) 中读取姿势样本数据，并创建 PoseClassifier 对象用于分类。
//处理姿势数据： getPoseResult(pose) 方法接受一个 Pose 对象作为输入，并返回一个包含分类结果和重复次数的字符串列表。
//平滑处理： 如果处于流模式 (isStreamMode)，使用 EMASmoothing 类对分类结果进行指数移动平均平滑，以减少噪声的影响。
//重复计数：
//维护一个 currentState 变量，跟踪当前姿势状态 (例如 DOWN、UP)。
//根据分类结果和当前状态更新状态机，判断是否完成了一个重复动作。
//如果完成了一个重复动作，则将 repCount 加 1，并通过 RepCountListener 接口通知外界。
//代码解读：
//POSE_SAMPLES_FILE：存储姿势样本数据的文件路径。
//PUSHUPS_CLASS，SQUATS_CLASS：定义了要进行计数的姿势类别。
//PoseState：枚举类型，表示不同的姿势状态。
//currentState：当前姿势状态。
//repCount：重复动作次数。
//isStreamMode：是否处于流模式，即处理连续的姿势数据。
//emaSmoothing：用于平滑分类结果的 EMA 对象。
//repCounters：存储每个姿势类别的 RepetitionCounter 对象的列表。
//poseClassifier：用于分类姿势的 PoseClassifier 对象。
//lastRepResult：上一次计数结果的字符串。
//loadPoseSamples(context)：从文件中加载姿势样本数据，并创建 PoseClassifier 对象。
//getPoseResult(pose)：处理输入的 Pose 对象，并返回分类结果和重复次数的字符串列表。
//RepCountListener：用于通知外界重复次数更新的接口。
//代码应用：
//这个代码可以用于各种需要进行姿势分类和计数的应用，例如：
//健身应用: 跟踪用户的健身进度，并提供实时反馈。
//运动分析: 分析运动员的动作，计算训练量。
//康复训练: 帮助患者进行康复训练，并监测训练效果。
//总结：
//PoseClassifierProcessor 类结合了姿势分类和重复计数的功能，可以有效地分析连续的姿势数据，并提供有用的信息。它可以作为构建健身、运动分析和康复训练等应用的基础。