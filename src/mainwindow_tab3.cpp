#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "SiftCore.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <QApplication>

#include <chrono>
#include <cmath>

using Clock = std::chrono::high_resolution_clock;

// Tab 3 only: ROI handling, matcher selection, and projection rendering.

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

    ui->btnMatchFeatures->setEnabled(false);
    QApplication::processEvents();

    const std::vector<QRect> rois = roiLabel ? roiLabel->getSelectedROIs() : std::vector<QRect>{};
    const float ratio = p3_ratioThresh;
    const int methodIndex = ui->matchMethodCombo ? ui->matchMethodCombo->currentIndex() : 0;
    const cv::Mat img1_copy = img1.clone();
    const cv::Mat img2_copy = img2.clone();
    const float contrastThresh = p2_contrastThresh;
    const std::vector<cv::KeyPoint> sift1_kp = siftResult1.keypoints;
    const cv::Mat sift1_desc = siftResult1.descriptors.clone();

    watcherMatch.setFuture(QtConcurrent::run([this, rois, ratio, methodIndex, img1_copy, img2_copy, contrastThresh, sift1_kp, sift1_desc]() {
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

            matchResult.matches = (methodIndex == 1) ? matchNCC(queryDesc, sift2_desc, ratio) : matchSSD(queryDesc, sift2_desc, ratio);
            matchResult.inliers = static_cast<int>(matchResult.matches.size());

            auto toBgr = [](const cv::Mat& src) {
                cv::Mat out;
                if (src.empty()) return out;
                if (src.channels() == 3) out = src.clone();
                else cv::cvtColor(src, out, cv::COLOR_GRAY2BGR);
                return out;
            };

            cv::Mat featurePreview = toBgr(siftResult1.annotated.empty() ? img1_copy : siftResult1.annotated);
            cv::Mat loadedPreview = toBgr(img2_copy);

            for (const QRect& roi : rois) {
                cv::rectangle(featurePreview, cv::Rect(roi.x(), roi.y(), std::max(1, roi.width()), std::max(1, roi.height())), cv::Scalar(46, 204, 113), 2, cv::LINE_AA);
            }

            for (const auto& m : matchResult.matches) {
                if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(queryKp.size())) continue;
                if (m.trainIdx < 0 || m.trainIdx >= static_cast<int>(sift2_kp.size())) continue;
                cv::circle(featurePreview, queryKp[m.queryIdx].pt, 4, cv::Scalar(255, 170, 60), 2, cv::LINE_AA);
                cv::circle(loadedPreview, sift2_kp[m.trainIdx].pt, 4, cv::Scalar(255, 170, 60), 2, cv::LINE_AA);
            }

            matchResult.composite = featurePreview;
            matchResult.loadedAnnotated = loadedPreview;
        } catch (...) {
            matchResult.composite = cv::Mat();
            matchResult.loadedAnnotated = cv::Mat();
            matchResult.inliers = 0;
            matchResult.matches.clear();
        }

        matchResult.timeMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }));
}

void MainWindow::onMatchingFinished()
{
    if (!matchResult.loadedAnnotated.empty()) ui->matchDisplayLoaded->setImage(matToPixmap(matchResult.loadedAnnotated));
    else if (!img2.empty()) ui->matchDisplayLoaded->setImage(matToPixmap(img2));

    if (roiLabel) {
        if (!matchResult.composite.empty()) roiLabel->setPixmap(matToPixmap(matchResult.composite));
        else roiLabel->setText("Matching failed — check console.");
    }

    const QString methodName = (ui->matchMethodCombo && ui->matchMethodCombo->currentIndex() == 1) ? "NCC" : "SSD";
    ui->matchInfo->setText(
        QString("%1  |  Matches: %2  |  Inliers: %3  |  Time: %4 ms")
            .arg(methodName)
            .arg(matchResult.matches.size())
            .arg(matchResult.inliers)
            .arg(matchResult.timeMs, 0, 'f', 1));

    setStatus(QString("Feature matching — %1 matches in %2 ms").arg(matchResult.matches.size()).arg(matchResult.timeMs, 0, 'f', 1));
    ui->btnMatchFeatures->setEnabled(true);
}
