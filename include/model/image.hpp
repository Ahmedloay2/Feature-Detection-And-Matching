#pragma once

/**
 * @file image.hpp
 * @brief Core image data container with typed pipeline cache.
 *
 * SOLID notes
 * ─────────────
 * SRP  – Image only owns data; processors operate on it externally.
 * OCP  – New stages can be added without touching this header.
 * LSP  – No inheritance here; kept as a plain value type.
 * ISP  – Thin interface; callers only call what they need.
 * DIP  – Processors depend on this abstraction, not on concrete types.
 *
 * Performance notes
 * ─────────────────
 * • Cache uses string_view-keyed lookup where possible (C++17).
 * • cv::Mat is ref-counted; store() avoids deep copies.
 * • clearCache() releases all intermediate memory in one shot.
 */

#include <opencv2/core.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <stdexcept>
#include <optional>

struct Image
{
    // ── data ─────────────────────────────────────────────────────────────────
    cv::Mat mat;  ///< Source image (CV_8UC3 BGR)

    // ── cache API ─────────────────────────────────────────────────────────────

    /// Store an intermediate result.  Mat header is copied; pixel data is shared.
    void store(std::string_view name, const cv::Mat& result)
    {
        cache_[std::string(name)] = result;
    }

    /// Move-store to avoid ref-count churn when the caller no longer needs the mat.
    void store(std::string_view name, cv::Mat&& result)
    {
        cache_[std::string(name)] = std::move(result);
    }

    /// Retrieve a stage.  Throws only on programmer error (missing stage).
    [[nodiscard]] const cv::Mat& get(std::string_view name) const
    {
        auto it = cache_.find(std::string(name));
        if (it == cache_.end())
            throw std::runtime_error("Pipeline stage not found: " + std::string(name));
        return it->second;
    }

    /// Non-throwing lookup — prefer this in hot paths.
    [[nodiscard]] std::optional<std::reference_wrapper<const cv::Mat>>
        tryGet(std::string_view name) const noexcept
    {
        auto it = cache_.find(std::string(name));
        if (it == cache_.end()) return std::nullopt;
        return std::cref(it->second);
    }

    [[nodiscard]] bool has(std::string_view name) const noexcept
    {
        return cache_.count(std::string(name)) != 0u;
    }

    void clearCache() noexcept { cache_.clear(); }

private:
    std::unordered_map<std::string, cv::Mat> cache_;
};