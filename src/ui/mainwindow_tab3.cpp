/**
 * @file mainwindow_tab3.cpp
 * @brief Implements Tab 3: Feature matching between two images with per-ROI filtering.
 *
 * Handles:
 * - Async descriptor matching (SSD with Lowe's ratio test or NCC with correlation)
 * - Per-ROI matching: each drawn rectangle is matched independently with a distinct color
 * - ROI coordinate clamping: ROIs are always intersected with img1 bounds before use
 * - Method selection and threshold tuning (SSD ratio threshold or NCC correlation)
 * - Match rendering: draws lines connecting corresponding keypoints per-ROI color
 * - Result statistics: displays total match count, per-ROI counts, and timing
 * - Interactive ROI management: add/undo/clear region selections
 */

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include "core/SiftCore.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <QApplication>

#include <chrono>
#include <cmath>

using Clock = std::chrono::high_resolution_clock;

// ─── Per-ROI colour palette (BGR) ─────────────────────────────────────────────
static const cv::Scalar kRoiColors[] = {
    {46, 204, 113}, // green
    {60, 60, 255},  // red
    {255, 165, 0},  // blue
    {0, 215, 255},  // yellow
    {255, 60, 200}, // magenta
    {0, 200, 200},  // olive
    {200, 100, 0},  // teal
    {128, 0, 255},  // violet
};
static constexpr int kNumRoiColors = static_cast<int>(std::size(kRoiColors));

// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::onMatchRatioChanged(int v)
{
    p3_ratioThresh = v / 100.0f;
    ui->matchRatioVal->setText(QString::number(p3_ratioThresh, 'f', 2));
}

void MainWindow::onClearROI()
{
    if (roiLabel)
        roiLabel->clearROI();
    ui->matchROIInfo->setText("ROI cleared — draw a new one.");
}

void MainWindow::onRemoveLastROI()
{
    if (roiLabel)
        roiLabel->removeLastROI();
    auto rois = roiLabel ? roiLabel->getSelectedROIs() : std::vector<QRect>{};
    ui->matchROIInfo->setText(
        rois.empty()
            ? "All ROIs removed — draw a new one."
            : QString("ROIs: %1").arg(rois.size()));
}

// ─── Descriptor matchers ──────────────────────────────────────────────────────
std::vector<cv::DMatch> MainWindow::matchSSD(const cv::Mat &desc1,
                                             const cv::Mat &desc2,
                                             float ratio) const
{
    std::vector<cv::DMatch> good;
    if (desc1.empty() || desc2.empty())
        return good;

    for (int i = 0; i < desc1.rows; ++i)
    {
        const float *q = desc1.ptr<float>(i);
        float best1 = 1e30f, best2 = 1e30f;
        int idx1 = -1;

        for (int j = 0; j < desc2.rows; ++j)
        {
            const float *t = desc2.ptr<float>(j);
            float ssd = 0.f;
            for (int d = 0; d < 128; ++d)
            {
                const float diff = q[d] - t[d];
                ssd += diff * diff;
            }
            if (ssd < best1)
            {
                best2 = best1;
                best1 = ssd;
                idx1 = j;
            }
            else if (ssd < best2)
                best2 = ssd;
        }

        if (idx1 >= 0 && std::sqrt(best1) < ratio * std::sqrt(best2))
            good.emplace_back(i, idx1, std::sqrt(best1));
    }
    return good;
}

std::vector<cv::DMatch> MainWindow::matchNCC(const cv::Mat &desc1,
                                             const cv::Mat &desc2,
                                             float minCorr) const
{
    std::vector<cv::DMatch> good;
    if (desc1.empty() || desc2.empty())
        return good;

    for (int i = 0; i < desc1.rows; ++i)
    {
        const float *q = desc1.ptr<float>(i);
        float qMean = 0.f;
        for (int d = 0; d < 128; ++d)
            qMean += q[d];
        qMean /= 128.f;

        float qNorm = 0.f;
        for (int d = 0; d < 128; ++d)
        {
            const float v = q[d] - qMean;
            qNorm += v * v;
        }
        if (qNorm <= 1e-12f)
            continue;
        qNorm = std::sqrt(qNorm);

        float best = -2.f;
        int bestIdx = -1;

        for (int j = 0; j < desc2.rows; ++j)
        {
            const float *t = desc2.ptr<float>(j);
            float tMean = 0.f;
            for (int d = 0; d < 128; ++d)
                tMean += t[d];
            tMean /= 128.f;

            float tNorm = 0.f, dot = 0.f;
            for (int d = 0; d < 128; ++d)
            {
                const float qv = q[d] - qMean;
                const float tv = t[d] - tMean;
                dot += qv * tv;
                tNorm += tv * tv;
            }
            if (tNorm <= 1e-12f)
                continue;

            const float ncc = dot / (qNorm * std::sqrt(tNorm));
            if (ncc > best)
            {
                best = ncc;
                bestIdx = j;
            }
        }

        if (bestIdx >= 0 && best >= minCorr)
            good.emplace_back(i, bestIdx, 1.0f - best);
    }
    return good;
}

// ─── onMatchFeatures ──────────────────────────────────────────────────────────
void MainWindow::onMatchFeatures()
{
    if (!siftResult1.valid)
    {
        setStatus("Run SIFT (Tab 2) first.");
        return;
    }
    if (img2.empty())
    {
        setStatus("Load Image 2 first.");
        return;
    }

    const std::vector<QRect> roisRaw =
        roiLabel ? roiLabel->getSelectedROIs() : std::vector<QRect>{};
    if (roisRaw.empty())
    {
        setStatus("Draw at least one ROI on the left image first.");
        return;
    }

    ui->btnMatchFeatures->setEnabled(false);
    QApplication::processEvents();

    const float ratio = p3_ratioThresh;
    const int selectedMode = ui->matchMethodCombo ? ui->matchMethodCombo->currentIndex() : 0;
    const cv::Mat img1_copy = img1.clone();
    const cv::Mat img2_copy = img2.clone();
    const float contrastThresh = p2_contrastThresh;
    const std::vector<cv::KeyPoint> sift1_kp = siftResult1.keypoints;
    const cv::Mat sift1_desc = siftResult1.descriptors.clone();

    // ── Clamp every ROI to img1 pixel bounds ────────────────────────────────
    // After matching the composite (img1|img2) is shown in roiLabel.
    // getSelectedROIs() returns coords in that pixmap's pixel space.
    // The left half of the composite == img1, so x ∈ [0, img1.cols).
    // ROIs drawn on the right half (img2 side) are silently discarded here.
    const QRect img1Bounds(0, 0, img1_copy.cols, img1_copy.rows);
    std::vector<QRect> rois;
    rois.reserve(roisRaw.size());
    for (const QRect &r : roisRaw)
    {
        const QRect clamped = r.intersected(img1Bounds);
        if (clamped.width() > 2 && clamped.height() > 2)
            rois.push_back(clamped);
    }

    if (rois.empty())
    {
        setStatus("All ROIs are outside the left (source) image — draw on the left half.");
        ui->matchROIInfo->setText("ROIs outside img1 — draw on the left half.");
        ui->btnMatchFeatures->setEnabled(true);
        return;
    }

    watcherMatch.setFuture(QtConcurrent::run(
        [this, rois, ratio, selectedMode,
         img1_copy, img2_copy, contrastThresh,
         sift1_kp, sift1_desc]()
        {
            const auto t0 = Clock::now();
            try
            {
                // ── Ensure img2 SIFT descriptors ────────────────────────────────
                std::vector<cv::KeyPoint> sift2_kp;
                cv::Mat sift2_desc;

                const bool sift2CacheOk =
                    siftResult2.valid &&
                    !siftResult2.descriptors.empty() &&
                    siftResult2.contrastThreshold == contrastThresh &&
                    siftResult2.numOctaves == p2_numOctaves &&
                    siftResult2.numScales == p2_numScales;

                if (!sift2CacheOk)
                {
                    cv_assign::SiftProcessor::extractFeatures(
                        img2_copy, sift2_kp, sift2_desc,
                        contrastThresh, p2_numOctaves, p2_numScales);
                    siftResult2.contrastThreshold = contrastThresh;
                    siftResult2.numOctaves = p2_numOctaves;
                    siftResult2.numScales = p2_numScales;
                }
                else
                {
                    sift2_kp = siftResult2.keypoints;
                    sift2_desc = siftResult2.descriptors.clone();
                }
                siftResult2.keypoints = sift2_kp;
                siftResult2.descriptors = sift2_desc;
                siftResult2.valid = true;

                // ── Build composite canvas ────────────────────────────────────
                auto toBgr = [](const cv::Mat &src)
                {
                    cv::Mat out;
                    if (src.empty())
                        return out;
                    if (src.channels() == 3)
                        out = src.clone();
                    else
                        cv::cvtColor(src, out, cv::COLOR_GRAY2BGR);
                    return out;
                };

                cv::Mat featureBase = toBgr(
                    siftResult1.annotated.empty() ? img1_copy : siftResult1.annotated);
                cv::Mat loadedBase = toBgr(img2_copy);

                // Scale img2 to fit the right half of the canvas (same height as img1)
                float rightScale = 1.0f;
                int rightOffX = featureBase.cols;
                int rightOffY = 0;

                cv::Mat rightPanel = cv::Mat::zeros(featureBase.rows, featureBase.cols, CV_8UC3);
                if (!loadedBase.empty())
                {
                    const float sx =
                        static_cast<float>(featureBase.cols) / static_cast<float>(loadedBase.cols);
                    const float sy =
                        static_cast<float>(featureBase.rows) / static_cast<float>(loadedBase.rows);
                    rightScale = std::min(sx, sy);

                    const int fitW = std::max(1, static_cast<int>(std::round(loadedBase.cols * rightScale)));
                    const int fitH = std::max(1, static_cast<int>(std::round(loadedBase.rows * rightScale)));

                    cv::Mat fitted;
                    cv::resize(loadedBase, fitted, cv::Size(fitW, fitH), 0, 0, cv::INTER_AREA);

                    const int localX = (featureBase.cols - fitW) / 2;
                    const int localY = (featureBase.rows - fitH) / 2;
                    fitted.copyTo(rightPanel(cv::Rect(localX, localY, fitW, fitH)));

                    rightOffX = featureBase.cols + localX;
                    rightOffY = localY;
                }

                // Final side-by-side canvas
                const int canvasH = featureBase.rows;
                const int canvasW = featureBase.cols * 2;
                cv::Mat composite = cv::Mat::zeros(canvasH, canvasW, CV_8UC3);
                featureBase.copyTo(composite(cv::Rect(0, 0, featureBase.cols, featureBase.rows)));
                rightPanel.copyTo(composite(cv::Rect(featureBase.cols, 0,
                                                     rightPanel.cols, rightPanel.rows)));

                // ── Per-ROI matching ──────────────────────────────────────────
                std::vector<cv::DMatch> allMatches;
                std::vector<cv::KeyPoint> allQueryKp;
                std::vector<int> roiMatchCounts; // per-ROI inlier count
                int totalInliers = 0;
                double totalMatchMs = 0.0;

                for (int ri = 0; ri < static_cast<int>(rois.size()); ++ri)
                {
                    const QRect &roi = rois[ri];
                    const cv::Scalar &color = kRoiColors[ri % kNumRoiColors];

                    // Draw ROI rectangle on composite (left half)
                    cv::rectangle(composite,
                                  cv::Rect(roi.x(), roi.y(),
                                           std::max(1, roi.width()),
                                           std::max(1, roi.height())),
                                  color, 2, cv::LINE_AA);

                    // ── Collect keypoints inside this ROI ──────────────────
                    std::vector<cv::KeyPoint> roiKp;
                    std::vector<int> roiIdx;
                    for (int i = 0; i < static_cast<int>(sift1_kp.size()); ++i)
                    {
                        const cv::Point2f &pt = sift1_kp[i].pt;
                        if (roi.contains(static_cast<int>(pt.x), static_cast<int>(pt.y)))
                        {
                            roiKp.push_back(sift1_kp[i]);
                            roiIdx.push_back(i);
                        }
                    }

                    if (roiKp.empty())
                    {
                        roiMatchCounts.push_back(0);
                        continue; // No keypoints in this ROI — skip
                    }

                    // ── Build per-ROI descriptor sub-matrix ───────────────
                    cv::Mat roiDesc(static_cast<int>(roiIdx.size()), 128, CV_32F);
                    for (int i = 0; i < static_cast<int>(roiIdx.size()); ++i)
                        sift1_desc.row(roiIdx[i]).copyTo(roiDesc.row(i));

                    // ── Run matching ───────────────────────────────────────
                    const auto tMatch0 = Clock::now();
                    std::vector<cv::DMatch> roiMatches;
                    if (selectedMode == 0)
                        roiMatches = matchSSD(roiDesc, sift2_desc, ratio);
                    else
                        roiMatches = matchNCC(roiDesc, sift2_desc, ratio);
                    totalMatchMs +=
                        std::chrono::duration<double, std::milli>(Clock::now() - tMatch0).count();

                    roiMatchCounts.push_back(static_cast<int>(roiMatches.size()));
                    totalInliers += static_cast<int>(roiMatches.size());

                    // ── Draw match lines on composite ──────────────────────
                    for (const auto &m : roiMatches)
                    {
                        if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(roiKp.size()))
                            continue;
                        if (m.trainIdx < 0 || m.trainIdx >= static_cast<int>(sift2_kp.size()))
                            continue;

                        const cv::Point2f pt1 = roiKp[m.queryIdx].pt;
                        const cv::Point2f pt2 = sift2_kp[m.trainIdx].pt;
                        const cv::Point2f pt2s(pt2.x * rightScale + static_cast<float>(rightOffX),
                                               pt2.y * rightScale + static_cast<float>(rightOffY));

                        cv::line(composite, pt1, pt2s, color, 1, cv::LINE_AA);
                        cv::circle(composite, pt1, 3, color, -1, cv::LINE_AA);
                        cv::circle(composite, pt2s, 3, color, -1, cv::LINE_AA);
                    }

                    // ── Accumulate for matchResult (adjust queryIdx offset) ──
                    const int offset = static_cast<int>(allQueryKp.size());
                    for (auto m : roiMatches)
                    {
                        m.queryIdx += offset;
                        allMatches.push_back(m);
                    }
                    allQueryKp.insert(allQueryKp.end(), roiKp.begin(), roiKp.end());
                }

                // ── Bail if no keypoints matched any ROI ─────────────────────
                if (totalInliers == 0 && allQueryKp.empty())
                {
                    // Store an informative composite that still shows the ROI boxes
                    matchResult.composite = composite;
                    matchResult.loadedAnnotated = loadedBase.clone();
                    matchResult.matches.clear();
                    matchResult.inliers = 0;
                    matchResult.timeMs =
                        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

                    if (selectedMode == 0)
                    {
                        p3_ssdMatches.clear();
                        p3_ssdInliers = 0;
                        p3_ssdTimeMs = totalMatchMs;
                        p3_featureSSD = composite;
                        p3_loadedSSD = loadedBase.clone();
                        p3_lastComputedMode = 0;
                    }
                    else
                    {
                        p3_nccMatches.clear();
                        p3_nccInliers = 0;
                        p3_nccTimeMs = totalMatchMs;
                        p3_featureNCC = composite;
                        p3_loadedNCC = loadedBase.clone();
                        p3_lastComputedMode = 1;
                    }
                    return;
                }

                // ── Build per-ROI summary string for status bar ───────────────
                // (stored in matchResult.timeMs field; real status built in finished())
                // We reuse the existing fields; per-ROI info goes into status string.
                if (selectedMode == 0)
                {
                    p3_ssdMatches = allMatches;
                    p3_ssdInliers = totalInliers;
                    p3_ssdTimeMs = totalMatchMs;
                    p3_featureSSD = composite;
                    p3_loadedSSD = loadedBase.clone();
                    p3_nccMatches.clear();
                    p3_nccInliers = 0;
                    p3_nccTimeMs = 0.0;
                    p3_featureNCC = cv::Mat();
                    p3_loadedNCC = cv::Mat();
                    p3_lastComputedMode = 0;

                    matchResult.composite = composite;
                    matchResult.loadedAnnotated = loadedBase.clone();
                    matchResult.matches = allMatches;
                    matchResult.inliers = totalInliers;
                }
                else
                {
                    p3_nccMatches = allMatches;
                    p3_nccInliers = totalInliers;
                    p3_nccTimeMs = totalMatchMs;
                    p3_featureNCC = composite;
                    p3_loadedNCC = loadedBase.clone();
                    p3_ssdMatches.clear();
                    p3_ssdInliers = 0;
                    p3_ssdTimeMs = 0.0;
                    p3_featureSSD = cv::Mat();
                    p3_loadedSSD = cv::Mat();
                    p3_lastComputedMode = 1;

                    matchResult.composite = composite;
                    matchResult.loadedAnnotated = loadedBase.clone();
                    matchResult.matches = allMatches;
                    matchResult.inliers = totalInliers;
                }
            }
            catch (...)
            {
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

            matchResult.timeMs =
                std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        }));
}

// ─── onMatchingFinished ───────────────────────────────────────────────────────
void MainWindow::onMatchingFinished()
{
    const int modeIdx =
        ui->matchMethodCombo ? ui->matchMethodCombo->currentIndex() : 0;

    if (modeIdx == 0 && p3_lastComputedMode == 0)
    {
        if (!p3_featureSSD.empty())
            matchResult.composite = p3_featureSSD;
        if (!p3_loadedSSD.empty())
            matchResult.loadedAnnotated = p3_loadedSSD;
        matchResult.matches = p3_ssdMatches;
        matchResult.inliers = p3_ssdInliers;
        matchResult.timeMs = p3_ssdTimeMs;
    }
    else if (modeIdx == 1 && p3_lastComputedMode == 1)
    {
        if (!p3_featureNCC.empty())
            matchResult.composite = p3_featureNCC;
        if (!p3_loadedNCC.empty())
            matchResult.loadedAnnotated = p3_loadedNCC;
        matchResult.matches = p3_nccMatches;
        matchResult.inliers = p3_nccInliers;
        matchResult.timeMs = p3_nccTimeMs;
    }
    else
    {
        ui->matchInfo->setText("Mode changed — press Run Match again.");
        ui->btnMatchFeatures->setEnabled(true);
        return;
    }

    if (roiLabel)
    {
        if (!matchResult.composite.empty())
        {
            roiLabel->setPixmap(matToPixmap(matchResult.composite));
            // The composite already has ROI boxes baked into its pixels.
            // Clear the InteractiveLabel overlay to avoid drawing them twice
            // (baked image + overlay = double box with a slight scale offset).
            // Arm the flag so the very next mouse-press wipes everything and
            // starts fresh, while still allowing multiple ROIs to accumulate
            // before the next Run Match.
            roiLabel->clearROI();
            roiLabel->setResetOnNextDraw(true);
            ui->matchROIInfo->setText("Match done — draw a new ROI to run again.");
        }
        else
            roiLabel->setText("Matching failed — check console.");
    }

    // ── Status: total + mode ────────────────────────────────────────────────
    const QString modeText = (modeIdx == 0) ? "SSD" : "NCC";
    ui->matchInfo->setText(
        QString("Mode:%1  |  Matches:%2  |  Inliers:%3  |  Time:%4 ms")
            .arg(modeText)
            .arg(matchResult.matches.size())
            .arg(matchResult.inliers)
            .arg(matchResult.timeMs, 0, 'f', 1));

    setStatus(QString("Feature matching done — %1:%2")
                  .arg(modeText)
                  .arg(matchResult.inliers));

    ui->btnMatchFeatures->setEnabled(true);
}