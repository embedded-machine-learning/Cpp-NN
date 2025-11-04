#pragma once

#include <limits>
#include <utility>

#define ENABLE_PRINT true

namespace helpers {
/*
A class to wrap a type and count the number of operations, neatly split into a type and a counter,
only copy can not be considered, as the translation of copy is not nessesseraly a copy operation
*/
// template <typename CounterType = long>
struct Benchmark {
    using CounterType = unsigned long;

  public:
    //
    static inline CounterType counted_multiplications{};
    static inline CounterType counted_additions{};
    static inline CounterType counted_divisions{};
    static inline CounterType counted_subtractions{};
    static inline CounterType counted_comparisons{};
    static inline CounterType counted_extractions{};
    static inline CounterType counted_abs{};
    static inline CounterType counted_sqrt{};


    static void resetAll() {
        counted_multiplications = 0;
        counted_additions       = 0;
        counted_divisions       = 0;
        counted_subtractions    = 0;
        counted_comparisons     = 0;
        counted_extractions     = 0;
        counted_abs             = 0;
        counted_sqrt            = 0;
    }

    template <typename Type>
    struct TypeInstance {
      private:
        Type value;

      public:
        static inline CounterType counted_multiplications{};
        static inline CounterType counted_additions{};
        static inline CounterType counted_divisions{};
        static inline CounterType counted_subtractions{};
        static inline CounterType counted_comparisons{};
        static inline CounterType counted_extractions{};
        static inline CounterType counted_abs{};
        static inline CounterType counted_sqrt{};

        using type                        = Type;
        constexpr TypeInstance() noexcept = default;
        constexpr TypeInstance(const Type &value) noexcept : value(value) {};

        template <typename OtherType>
        constexpr TypeInstance(const OtherType &value) noexcept : value(static_cast<Type>(value)) {
        }

        constexpr TypeInstance(const TypeInstance &other) noexcept : value(other.value) {
        }

        constexpr TypeInstance(TypeInstance &&other) noexcept : value(std::move(other.value)) {
        }

        Type get() const {
            return value;
        }

        static void resetAll() {
            counted_multiplications = 0;
            counted_additions       = 0;
            counted_divisions       = 0;
            counted_subtractions    = 0;
            counted_comparisons     = 0;
            counted_extractions     = 0;
            counted_abs             = 0;
            counted_sqrt            = 0;
        }

        // Static cast operators
        operator Type() const noexcept {
            counted_extractions++;
            Benchmark::counted_extractions++;
            return value;
        };

        // Assignment operators
        template <typename OtherType>
        TypeInstance &operator=(const OtherType &other) noexcept {
            value = static_cast<Type>(other);
            return *this;
        }

        template <typename OtherType>
        constexpr TypeInstance<Type> &operator=(const TypeInstance<OtherType> &other) noexcept {
            value = static_cast<Type>(other.value);
            return *this;
        }

        constexpr TypeInstance<Type> &operator=(const TypeInstance &other) noexcept {
            value = other.value;
            return *this;
        }

        template <typename OtherType>
        constexpr TypeInstance<Type> &operator=(TypeInstance<OtherType> &&other) noexcept {
            value = static_cast<Type>(other.value);
            return *this;
        }

        template <typename OtherType>
        TypeInstance<Type> &operator+=(const TypeInstance<OtherType> &other) noexcept {
            value += other.value;
            counted_additions++;
            Benchmark::counted_additions++;
            return *this;
        }

        template <typename OtherType>
        TypeInstance<Type> &operator-=(const TypeInstance<OtherType> &other) noexcept {
            value -= other.value;
            counted_subtractions++;
            Benchmark::counted_subtractions++;
            return *this;
        }

        template <typename OtherType>
        TypeInstance<Type> &operator*=(const TypeInstance<OtherType> &other) noexcept {
            value *= other.value;
            counted_multiplications++;
            Benchmark::counted_multiplications++;
            return *this;
        }

        template <typename OtherType>
        TypeInstance<Type> &operator/=(const TypeInstance<OtherType> &other) noexcept {
            value /= other.value;
            counted_divisions++;
            Benchmark::counted_divisions++;
            return *this;
        }

        // Math operators
        template <typename OtherType>
        TypeInstance<Type> operator+(const TypeInstance<OtherType> &other) const noexcept {
            TypeInstance<Type> ret = *this;
            ret += other;
            return ret;
        }

        template <typename OtherType>
        TypeInstance<Type> operator-(const TypeInstance<OtherType> &other) const noexcept {
            TypeInstance ret = *this;
            ret -= other;
            return ret;
        }

        template <typename OtherType>
        TypeInstance<Type> operator*(const TypeInstance<OtherType> &other) const noexcept {
            TypeInstance ret = *this;
            ret *= other;
            return ret;
        }

        template <typename OtherType>
        TypeInstance<Type> operator/(const TypeInstance<OtherType> &other) const noexcept {
            TypeInstance ret = *this;
            ret /= other;
            return ret;
        }

        // Comparison operators
        template <typename OtherType>
        bool operator==(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value == other.value;
        }

        template <typename OtherType>
        bool operator!=(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value != other.value;
        }

        template <typename OtherType>
        bool operator<(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value < other.value;
        }

        template <typename OtherType>
        bool operator>(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value > other.value;
        }

        template <typename OtherType>
        bool operator<=(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value <= other.value;
        }

        template <typename OtherType>
        bool operator>=(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value >= other.value;
        }
    };
};

} // namespace helpers

#include <cmath>

__attribute__((always_inline)) inline helpers::Benchmark::TypeInstance<float> fabsf(const helpers::Benchmark::TypeInstance<float> &value) {
    helpers::Benchmark::TypeInstance<float> ret(fabsf(value.get()));
    helpers::Benchmark::counted_abs++;
    helpers::Benchmark::TypeInstance<float>::counted_abs++;
    return ret;
}

__attribute__((always_inline)) inline helpers::Benchmark::TypeInstance<double> fabs(const helpers::Benchmark::TypeInstance<double> &value) {
    helpers::Benchmark::TypeInstance<double> ret(fabs(value.get()));
    helpers::Benchmark::counted_abs++;
    helpers::Benchmark::TypeInstance<double>::counted_abs++;
    return ret;
}

__attribute__((always_inline)) inline helpers::Benchmark::TypeInstance<float> sqrtf(const helpers::Benchmark::TypeInstance<float> &value) {
    helpers::Benchmark::TypeInstance<float> ret(sqrtf(value.get()));
    helpers::Benchmark::counted_sqrt++;
    helpers::Benchmark::TypeInstance<float>::counted_sqrt++;
    return ret;
}

__attribute__((always_inline)) inline helpers::Benchmark::TypeInstance<double> sqrt(const helpers::Benchmark::TypeInstance<double> &value) {
    helpers::Benchmark::TypeInstance<double> ret(sqrt(value.get()));
    helpers::Benchmark::counted_sqrt++;
    helpers::Benchmark::TypeInstance<double>::counted_sqrt++;
    return ret;
}

// Numeric limit
template <typename Type>
class std::numeric_limits<helpers::Benchmark::TypeInstance<Type>> : public std::numeric_limits<Type> {};