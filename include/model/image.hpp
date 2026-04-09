/**
 * @file image.hpp
 * @brief Defines the Image data container that caches intermediate processing results by name.
 */

#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <stdexcept>
#include <optional>

/// @brief Image container encapsulating the source image and a processing pipeline cache.
///
/// This class manages both the original image (in `mat`) and intermediate results
/// from processing stages (grayscale, gradient, structure_tensor, harris_response, etc.)
/// stored in a hash map. Allows processors to avoid redundant recomputation by caching.
///
/// **Usage Pattern:**
/// ```cpp
/// Image img;
/// img.mat = cv::imread("photo.jpg");       // Set source
/// convertToGrayscale(img);                 // Stores result in cache["grayscale"]
/// if (img.has("grayscale")) {              // Check if stage exists
///     cv::Mat gray = img.get("grayscale"); // Retrieve cached result
/// }
/// ```
struct Image
{
    // ── data ─────────────────────────────────────────────────────────────────
    cv::Mat mat;  ///< Source image (CV_8UC3 BGR or other formats)

    // ── cache API ─────────────────────────────────────────────────────────────

    /// @brief Store an intermediate processing result under a named key.
    /// Mat header is copied; pixel data is shared (reference counted).
    void store(std::string_view name, const cv::Mat& result)
    {
        cache_[std::string(name)] = result;
    }

    /// @brief Move-store variant to reduce reference-count churn.
    /// Use when the original Mat is no longer needed.
    void store(std::string_view name, cv::Mat&& result)
    {
        cache_[std::string(name)] = std::move(result);
    }

    /// @brief Retrieve a cached stage result by name.
    /// @param name Stage name (e.g., "grayscale", "gradient_xx")
    /// @return Const reference to the cached Mat
    /// @throws std::runtime_error if key not found
    [[nodiscard]] const cv::Mat& get(std::string_view name) const
    {
        auto it = cache_.find(std::string(name));
        if (it == cache_.end())
            throw std::runtime_error("Pipeline stage not found: " + std::string(name));
        return it->second;
    }

    /// @brief Non-throwing lookup returning an optional reference.
    /// Prefer this in performance-critical paths to avoid exceptions.
    /// @param name Stage name
    /// @return Optional const reference to the cached Mat, or std::nullopt if not found
    [[nodiscard]] std::optional<std::reference_wrapper<const cv::Mat>>
        tryGet(std::string_view name) const noexcept
    {
        auto it = cache_.find(std::string(name));
        if (it == cache_.end()) return std::nullopt;
        return std::cref(it->second);
    }

    /// @brief Check if a stage result is cached without retrieving it.
    /// @param name Stage name
    /// @return true if stage exists in cache, false otherwise
    [[nodiscard]] bool has(std::string_view name) const noexcept
    {
        return cache_.count(std::string(name)) != 0u;
    }

    /// @brief Clear all cached intermediate results.
    /// Frees memory but preserves the source image (mat).
    void clearCache() noexcept { cache_.clear(); }

private:
    std::unordered_map<std::string, cv::Mat> cache_;
};