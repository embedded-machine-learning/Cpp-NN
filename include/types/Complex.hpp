#pragma once

#include <array>

template <typename T>
class Complex {
  public:
    std::array<T, 2> _data;
    using value_type = T;

    constexpr Complex() : _data({0, 0}) {
    }

    constexpr Complex(T real) : _data({real, 0}) {
    }

    constexpr Complex(T real, T imag) : _data({real, imag}) {
    }

    constexpr Complex(const Complex<T> &c) : _data(c._data) {
    }

    constexpr Complex(Complex<T> &&c) = default;

    constexpr T real() const {
        return _data[0];
    }

    __attribute__((always_inline)) inline T &real() {
        return _data[0];
    }

    constexpr T imag() const {
        return _data[1];
    }

    __attribute__((always_inline)) inline T &imag() {
        return _data[1];
    }

    __attribute__((always_inline)) inline Complex<T> &operator=(const Complex<T> &other) {
        _data[0] = other._data[0];
        _data[1] = other._data[1];
        return *this;
    }

    __attribute__((always_inline)) inline Complex<T> &operator+=(const Complex<T> &other) {
        _data[0] += other._data[0];
        _data[1] += other._data[1];
        return *this;
    }

    __attribute__((always_inline)) inline Complex<T> &operator-=(const Complex<T> &other) {
        _data[0] -= other._data[0];
        _data[1] -= other._data[1];
        return *this;
    }

    __attribute__((always_inline)) inline Complex<T> &operator*=(const Complex<T> &other) {
        T real_part = _data[0] * other._data[0] - _data[1] * other._data[1];
        T imag_part = _data[0] * other._data[1] + _data[1] * other._data[0];
        _data[0]     = real_part;
        _data[1]     = imag_part;
        return *this;
    }

    __attribute__((always_inline)) inline bool operator==(const Complex<T> &other) const {
        return (_data[0] == other._data[0]) && (_data[1] == other._data[1]);
    }

    __attribute__((always_inline)) inline Complex<T> operator+(const Complex<T> &other) const {
        return Complex<T>(_data[0] + other._data[0], _data[1] + other._data[1]);
    }

    __attribute__((always_inline)) inline Complex<T> operator-(const Complex<T> &other) const {
        return Complex<T>(_data[0] - other._data[0], _data[1] - other._data[1]);
    }

    __attribute__((always_inline)) inline Complex<T> operator*(const Complex<T> &other) const {
        return Complex<T>(_data[0] * other._data[0] - _data[1] * other._data[1], _data[0] * other._data[1] + _data[1] * other._data[0]);
    }

    __attribute__((always_inline)) inline Complex<T> operator*(const T &scalar) const {
        return Complex<T>(_data[0] * scalar, _data[1] * scalar);
    }

    __attribute__((always_inline)) inline Complex<T> operator/(const T &scalar) const {
        return Complex<T>(_data[0] / scalar, _data[1] / scalar);
    }

    __attribute__((always_inline)) inline Complex<T> operator+(const T &scalar) const {
        return Complex<T>(_data[0] + scalar, _data[1]);
    }

    __attribute__((always_inline)) inline Complex<T> operator-(const T &scalar) const {
        return Complex<T>(_data[0] - scalar, _data[1]);
    }
};

template <typename T>
__attribute__((always_inline)) inline Complex<T> operator*(const T &scalar, const Complex<T> &c) {
    return Complex<T>(scalar * c.real(), scalar * c.imag());
}

template <typename T>
__attribute__((always_inline)) inline Complex<T> operator+(const T &scalar, const Complex<T> &c) {
    return Complex<T>(scalar + c.real(), c.imag());
}
