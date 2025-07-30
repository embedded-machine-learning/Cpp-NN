#pragma once

#include <limits>

#define ENABLE_PRINT true

namespace helpers {
/*
A class to wrap a type and count the number of operations, neatly split into a type and a counter,
only copy can not be considered, as the translation of copy is not nessesseraly a copy operation
*/
// template <typename CounterType = long>
struct Benchmark {
    using CounterType = long;

  public:
    //
    static inline CounterType counted_multiplications{};
    static inline CounterType counted_additions{};
    static inline CounterType counted_divisions{};
    static inline CounterType counted_subtractions{};
    static inline CounterType counted_comparisons{};
    static inline CounterType counted_extractions{};

    static void resetAll() {
        counted_multiplications = 0;
        counted_additions       = 0;
        counted_divisions       = 0;
        counted_subtractions    = 0;
        counted_comparisons     = 0;
        counted_extractions     = 0;
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

        using type                        = Type;
        constexpr TypeInstance() noexcept = default;
        constexpr TypeInstance(const Type &value) noexcept : value(value) {};
        template <typename OtherType>
        constexpr TypeInstance(const OtherType &value) noexcept : value(static_cast<Type>(value)) {};
        constexpr TypeInstance(const TypeInstance &other) noexcept : value(other.value) {};
        constexpr TypeInstance(TypeInstance &&other) noexcept : value(std::move(other.value)) {};

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
        };

        template <typename OtherType>
        TypeInstance<Type> &operator=(const TypeInstance<OtherType> &other) noexcept {
            value = static_cast<Type>(other.value);
            return *this;
        };
    
        TypeInstance<Type> &operator=(const TypeInstance &other) noexcept {
            value = other.value;
            return *this;
        };

        template <typename OtherType>
        TypeInstance<Type> &operator=(TypeInstance<OtherType> &&other) noexcept {
            value = static_cast<Type>(other.value);
            return *this;
        };

        template <typename OtherType>
        TypeInstance<Type> &operator+=(const TypeInstance<OtherType> &other) noexcept {
            value += other.value;
            counted_additions++;
            Benchmark::counted_additions++;
            return *this;
        };

        template <typename OtherType>
        TypeInstance<Type> &operator-=(const TypeInstance<OtherType> &other) noexcept {
            value -= other.value;
            counted_subtractions++;
            Benchmark::counted_subtractions++;
            return *this;
        };

        template <typename OtherType>
        TypeInstance<Type> &operator*=(const TypeInstance<OtherType> &other) noexcept {
            value *= other.value;
            counted_multiplications++;
            Benchmark::counted_multiplications++;
            return *this;
        };

        template <typename OtherType>
        TypeInstance<Type> &operator/=(const TypeInstance<OtherType> &other) noexcept {
            value /= other.value;
            counted_divisions++;
            Benchmark::counted_divisions++;
            return *this;
        };

        // Math operators
        template <typename OtherType>
        TypeInstance<Type> operator+(const TypeInstance<OtherType> &other) const noexcept {
            TypeInstance<Type> ret = *this;
            ret += other;
            return ret;
        };

        template <typename OtherType>
        TypeInstance<Type> operator-(const TypeInstance<OtherType> &other) const noexcept {
            TypeInstance ret = *this;
            ret -= other;
            return ret;
        };

        template <typename OtherType>
        TypeInstance<Type> operator*(const TypeInstance<OtherType> &other) const noexcept {
            TypeInstance ret = *this;
            ret *= other;
            return ret;
        };

        template <typename OtherType>
        TypeInstance<Type> operator/(const TypeInstance<OtherType> &other) const noexcept {
            TypeInstance ret = *this;
            ret /= other;
            return ret;
        };

        // Comparison operators
        template <typename OtherType>
        bool operator==(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value == other.value;
        };

        template <typename OtherType>
        bool operator!=(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value != other.value;
        };

        template <typename OtherType>
        bool operator<(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value < other.value;
        };

        template <typename OtherType>
        bool operator>(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value > other.value;
        };

        template <typename OtherType>
        bool operator<=(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value <= other.value;
        };

        template <typename OtherType>
        bool operator>=(const TypeInstance<OtherType> &other) const noexcept {
            counted_comparisons++;
            Benchmark::counted_comparisons++;
            return value >= other.value;
        };
    };
};

} // namespace helpers

// Print
#if ENABLE_PRINT
#include <iostream>
template <typename Type>
std::ostream &operator<<(std::ostream &os, const helpers::Benchmark::TypeInstance<Type> &obj) {
    os << obj.get();
    return os;
}
#endif

// Numeric limit
template <typename Type>
class std::numeric_limits<helpers::Benchmark::TypeInstance<Type>> : public std::numeric_limits<Type> {
}; 