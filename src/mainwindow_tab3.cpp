/**
 * @file mainwindow_tab3.cpp
 * @brief Implements Tab 3: Feature matching between two images with ROI filtering.
 *
 * Handles:
 * - Async descriptor matching (SSD with Lowe's ratio test or NCC with correlation)
 * - ROI-based filtering: restrict matching to user-drawn rectangular regions
 * - Method selection and threshold tuning (SSD ratio threshold or NCC correlation)
 * - Match rendering: draws lines connecting corresponding keypoints
 * - Result statistics: displays match count, computation time, match quality
 * - Interactive ROI management: add/undo/clear region selections
 */

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "SiftCore.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <QApplication>

#include <chrono>
#include <cmath>

using Clock = std::chrono::high_resolution_clock;

void MainWindow::onMatchRatioChanged(int v)
{
    p3_ratioThresh = v / 100.0f;
    ui->matchRatioVal->setText(QString::number(p3_ratioThresh, 'f', 2));
}

void MainWindow::onClearROI()
{
    if (roiLabel) roiLabel->clearROI();
    ui->matchROIInfo->setText("ROI cleared — draw a new one.");
}

void MainWindow::onRemoveLastROI()
{
    if (roiLabel) roiLabel->removeLastROI();
    auto rois = roiLabel ? roiLabel->getSelectedROIs() : std::vector<QRect>{};
    ui->matchROIInfo->setText(rois.empty() ? "All ROIs removed — draw a new one." : QString("ROIs: %1").arg(rois.size()));
}

std::vector<cv::DMatch> MainWindow::matchSSD(const cv::Mat& desc1, const cv::Mat& desc2, float ratio) const
{
    std::vector<cv::DMatch> good;
    if (desc1.empty() || desc2.empty()) return good;

    for (int i = 0; i < desc1.rows; ++i) {
        const float* q = desc1.ptr<float>(i);
        float best1 = 1e30f, best2 = 1e30f;
        int idx1 = -1;

        for (int j = 0; j < desc2.rows; ++j) {
            const float* t = desc2.ptr<float>(j);
            float ssd = 0.f;
            for (int d = 0; d < 128; ++d) {
                const float diff = q[d] - t[d];
                ssd += diff * diff;
            }
            if (ssd < best1) { best2 = best1; best1 = ssd; idx1 = j; }
            else if (ssd < best2) best2 = ssd;
        }

        if (idx1 >= 0 && std::sqrt(best1) < ratio * std::sqrt(best2)) {
            good.emplace_back(i, idx1, std::sqrt(best1));
        }
    }
    return good;
}

std::vector<cv::DMatch> MainWindow::matchNCC(const cv::Mat& desc1, const cv::Mat& desc2, float minCorr) const
{
    std::vector<cv::DMatch> good;
    if (desc1.empty() || desc2.empty()) return good;

    for (int i = 0; i < desc1.rows; ++i) {
        const float* q = desc1.ptr<float>(i);
        float qMean = 0.f;
        for (int d = 0; d < 128; ++d) qMean += q[d];
        qMean /= 128.f;

        float qNorm = 0.f;
        for (int d = 0; d < 128; ++d) {
            const float v = q[d] - qMean;
            qNorm += v * v;
        }
        if (qNorm <= 1e-12f) continue;
        qNorm = std::sqrt(qNorm);

        float best = -2.f;
        int bestIdx = -1;

        for (int j = 0; j < desc2.rows; ++j) {
            const float* t = desc2.ptr<float>(j);
            float tMean = 0.f;
            for (int d = 0; d < 128; ++d) tMean += t[d];
            tMean /= 128.f;

            float tNorm = 0.f;
            float dot = 0.f;
            for (int d = 0; d < 128; ++d) {
                const float qv = q[d] - qMean;
                const float tv = t[d] - tMean;
                dot += qv * tv;
                tNorm += tv * tv;
            }
            if (tNorm <= 1e-12f) continue;

            const float ncc = dot / (qNorm * std::sqrt(tNorm));
            if (ncc > best) {
                best = ncc;
                bestIdx = j;
            }
        }

        if (bestIdx >= 0 && best >= minCorr) {
            good.emplace_back(i, bestIdx, 1.0f - best);
        }
    }

    return good;
}

void MainWindow::onMatchFeatures()
{
    if (!siftResult1.valid) { setStatus("Run SIFT (Tab 2) first."); return; }
    if (img2.empty()) { setStatus("Load Image 2 first."); return; }

    const std::vector<QRect> rois = roiLabel ? roiLabel->getSelectedROIs() : std::vector<QRect>{};
    if (rois.empty()) { setStatus("Draw ROI first in the left image area."); return; }

    ui->btnMatchFeatures->setEnabled(false);
    QApplication::processEvents();

    const float ratio = p3_ratioThresh;
    const int selectedMode = ui->matchMethodCombo ? ui->matchMethodCombo->currentIndex() : 0;
    const cv::Mat img1_copy = img1.clone();
    const cv::Mat img2_copy = img2.clone();
    const float contrastThresh = p2_contrastThresh;
    const std::vector<cv::KeyPoint> sift1_kp = siftResult1.keypoints;
    const cv::Mat sift1_desc = siftResult1.descriptors.clone();

    watcherMatch.setFuture(QtConcurrent::run([this, rois, ratio, selectedMode, img1_copy, img2_copy, contrastThresh, sift1_kp, sift1_desc]() {
        const auto t0 = Clock::now();
        try {
            std::vector<cv::KeyPoint> sift2_kp;
            cv::Mat sift2_desc;
            const bool sift2CacheMatches = siftResult2.valid && !siftResult2.descriptors.empty() &&
                                           siftResult2.contrastThreshold == contrastThresh &&
                                           siftResult2.numOctaves == p2_numOctaves &&
                                           siftResult2.numScales == p2_numScales;
            if (!sift2CacheMatches) {
                cv_assign::SiftProcessor::extractFeatures(img2_copy, sift2_kp, sift2_desc, contrastThresh, p2_numOctaves, p2_numScales);
                siftResult2.contrastThreshold = contrastThresh;
                siftResult2.numOctaves = p2_numOctaves;
                siftResult2.numScales = p2_numScales;
            } else {
                sift2_kp = siftResult2.keypoints;
                sift2_desc = siftResult2.descriptors.clone();
            }
            siftResult2.keypoints = sift2_kp;
            siftResult2.descriptors = sift2_desc;
            siftResult2.valid = true;

            cv::Mat queryDesc = sift1_desc;
            std::vector<cv::KeyPoint> queryKp = sift1_kp;

            if (!rois.empty()) {
                std::vector<cv::KeyPoint> filtKp;
                std::vector<int> filtIdx;
                for (int i = 0; i < static_cast<int>(queryKp.size()); ++i) {
                    const cv::Point2f& pt = queryKp[i].pt;
                    for (const QRect& r : rois) {
                        if (r.contains(static_cast<int>(pt.x), static_cast<int>(pt.y))) {
                            filtKp.push_back(queryKp[i]);
                            filtIdx.push_back(i);
                            break;
                        }
                    }
                }
                if (!filtIdx.empty()) {
                    queryKp = filtKp;
                    queryDesc = cv::Mat(static_cast<int>(filtIdx.size()), 128, CV_32F);
                    for (int i = 0; i < static_cast<int>(filtIdx.size()); ++i) {
                        sift1_desc.row(filtIdx[i]).copyTo(queryDesc.row(i));
                    }
                }
            }

            p3_ssdMatches.clear();
            p3_nccMatches.clear();
            p3_ssdInliers = 0;
            p3_nccInliers = 0;
            p3_ssdTimeMs = 0.0;
            p3_nccTimeMs = 0.0;

            if (selectedMode == 0) {
                const auto tSSD0 = Clock::now();
                p3_ssdMatches = matchSSD(queryDesc, sift2_desc, ratio);
                p3_ssdTimeMs = std::chrono::duration<double, std::milli>(Clock::now() - tSSD0).count();
            } else {
                const auto tNCC0 = Clock::now();
                p3_nccMatches = matchNCC(queryDesc, sift2_desc, ratio);
                p3_nccTimeMs = std::chrono::duration<double, std::milli>(Clock::now() - tNCC0).count();
            }

            auto toBgr = [](const cv::Mat& src) {
                cv::Mat out;
                if (src.empty()) return out;
                if (src.channels() == 3) out = src.clone();
                else cv::cvtColor(src, out, cv::COLOR_GRAY2BGR);
                return out;
            };

            cv::Mat featureBase = toBgr(siftResult1.annotated.empty() ? img1_copy : siftResult1.annotated);
            cv::Mat loadedBase = toBgr(img2_copy);

            for (const QRect& roi : rois) {
                cv::rectangle(featureBase, cv::Rect(roi.x(), roi.y(), std::max(1, roi.width()), std::max(1, roi.height())), cv::Scalar(46, 204, 113), 2, cv::LINE_AA);
            }

            auto drawCrossImageLines = [&](const std::vector<cv::DMatch>& matches,
                                           cv::Mat& featureOut,
                                           cv::Mat& loadedOut,
                                           const cv::Scalar& color,
                                           int& inlierOut) {
                cv::Mat left = featureBase.clone();
                cv::Mat right = cv::Mat::zeros(left.rows, left.cols, CV_8UC3);

                // Keep a fixed split layout: image1 occupies left half, right half is either
                // empty or contains image2 fitted to the same half height.
                float rightScale = 1.0f;
                int rightOffX = left.cols;
                int rightOffY = 0;
                if (!loadedBase.empty() && loadedBase.cols > 0 && loadedBase.rows > 0) {
                    const float sx = static_cast<float>(left.cols) / static_cast<float>(loadedBase.cols);
                    const float sy = static_cast<float>(left.rows) / static_cast<float>(loadedBase.rows);
                    rightScale = std::min(sx, sy);

                    const int fitW = std::max(1, static_cast<int>(std::round(loadedBase.cols * rightScale)));
                    const int fitH = std::max(1, static_cast<int>(std::round(loadedBase.rows * rightScale)));

                    cv::Mat fitted;
                    cv::resize(loadedBase, fitted, cv::Size(fitW, fitH), 0, 0, cv::INTER_AREA);

                    const int localX = (left.cols - fitW) / 2;
                    const int localY = (left.rows - fitH) / 2;
                    fitted.copyTo(right(cv::Rect(localX, localY, fitW, fitH)));
                    rightOffX = left.cols + localX;
                    rightOffY = localY;
                }

                const int canvasH = left.rows;
                const int canvasW = left.cols * 2;
                featureOut = cv::Mat::zeros(canvasH, canvasW, CV_8UC3);
                left.copyTo(featureOut(cv::Rect(0, 0, left.cols, left.rows)));
                right.copyTo(featureOut(cv::Rect(left.cols, 0, right.cols, right.rows)));
                loadedOut = loadedBase.clone();
                inlierOut = 0;

                for (const auto& m : matches) {
                    if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(queryKp.size())) continue;
                    if (m.trainIdx < 0 || m.trainIdx >= static_cast<int>(sift2_kp.size())) continue;

                    const cv::Point2f pt1 = queryKp[m.queryIdx].pt;       // in image 1 (left)
                    const cv::Point2f pt2 = sift2_kp[m.trainIdx].pt;      // in image 2 (right)
                    const cv::Point2f pt2Shifted(pt2.x * rightScale + rightOffX,
                                                 pt2.y * rightScale + rightOffY);

                    ++inlierOut;
                    cv::line(featureOut, pt1, pt2Shifted, color, 2, cv::LINE_AA);
                    cv::circle(featureOut, pt1, 3, color, -1, cv::LINE_AA);
                    cv::circle(featureOut, pt2Shifted, 3, color, -1, cv::LINE_AA);
                    cv::circle(loadedOut, pt2, 3, color, 1, cv::LINE_AA);
                }
            };

            p3_featureSSD = featureBase.clone();
            p3_featureNCC = featureBase.clone();
            p3_loadedSSD = loadedBase.clone();
            p3_loadedNCC = loadedBase.clone();

            if (selectedMode == 0) {
                drawCrossImageLines(p3_ssdMatches, p3_featureSSD, p3_loadedSSD, cv::Scalar(60, 190, 255), p3_ssdInliers);
                matchResult.composite = p3_featureSSD;
                matchResult.loadedAnnotated = p3_loadedSSD;
                matchResult.matches = p3_ssdMatches;
                matchResult.inliers = p3_ssdInliers;
                matchResult.timeMs = p3_ssdTimeMs;
                p3_lastComputedMode = 0;
            } else {
                drawCrossImageLines(p3_nccMatches, p3_featureNCC, p3_loadedNCC, cv::Scalar(255, 120, 120), p3_nccInliers);
                matchResult.composite = p3_featureNCC;
                matchResult.loadedAnnotated = p3_loadedNCC;
                matchResult.matches = p3_nccMatches;
                matchResult.inliers = p3_nccInliers;
                matchResult.timeMs = p3_nccTimeMs;
                p3_lastComputedMode = 1;
            }
        } catch (...) {
            matchResult.composite = cv::Mat();
            matchResult.loadedAnnotated = cv::Mat();
            matchResult.inliers = 0;
            matchResult.matches.clear();
            p3_ssdMatches.clear();
            p3_nccMatches.clear();
            p3_lastComputedMode = -1;
            p3_ssdInliers = 0;
            p3_nccInliers = 0;
            p3_ssdTimeMs = 0.0;
            p3_nccTimeMs = 0.0;
            p3_featureSSD = cv::Mat();
            p3_featureNCC = cv::Mat();
            p3_loadedSSD = cv::Mat();
            p3_loadedNCC = cv::Mat();
        }

        matchResult.timeMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }));
}

void MainWindow::onMatchingFinished()
{
    const int modeIdx = ui->matchMethodCombo ? ui->matchMethodCombo->currentIndex() : 0;

    if (modeIdx == 0 && p3_lastComputedMode == 0) {
        if (!p3_featureSSD.empty()) matchResult.composite = p3_featureSSD;
        if (!p3_loadedSSD.empty()) matchResult.loadedAnnotated = p3_loadedSSD;
        matchResult.matches = p3_ssdMatches;
        matchResult.inliers = p3_ssdInliers;
        matchResult.timeMs = p3_ssdTimeMs;
    } else if (modeIdx == 1 && p3_lastComputedMode == 1) {
        if (!p3_featureNCC.empty()) matchResult.composite = p3_featureNCC;
        if (!p3_loadedNCC.empty()) matchResult.loadedAnnotated = p3_loadedNCC;
        matchResult.matches = p3_nccMatches;
        matchResult.inliers = p3_nccInliers;
        matchResult.timeMs = p3_nccTimeMs;
    } else {
        ui->matchInfo->setText("Mode changed — press Run Match");
        ui->btnMatchFeatures->setEnabled(true);
        return;
    }

    if (roiLabel) {
        if (!matchResult.composite.empty()) roiLabel->setPixmap(matToPixmap(matchResult.composite));
        else roiLabel->setText("Matching failed — check console.");
    }

    QString modeText = (modeIdx == 0) ? "SSD" : "NCC";

    ui->matchInfo->setText(
        QString("Mode:%1  |  Matches:%2  |  Inliers:%3  |  Time:%4 ms")
            .arg(modeText)
            .arg(matchResult.matches.size())
            .arg(matchResult.inliers)
            .arg(matchResult.timeMs, 0, 'f', 1));

    setStatus(QString("Feature matching done — %1:%2")
                  .arg(modeText)
                  .arg(matchResult.matches.size()));
    ui->btnMatchFeatures->setEnabled(true);
}
