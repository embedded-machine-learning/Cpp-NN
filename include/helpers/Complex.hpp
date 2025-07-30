#pragma once

template <typename T>
class Complex {
private:
    T _real;
    T _imag;

public:
    using value_type = T;
    
    constexpr Complex() : _real(0), _imag(0) {}
    constexpr Complex(T real) : _real(real), _imag(0) {}
    constexpr Complex(T real, T imag) : _real(real), _imag(imag) {}
    constexpr Complex (const Complex &c) : _real(c._real), _imag(c._imag) {}
    constexpr Complex (Complex &&c) : _real(std::move(c._real)), _imag(std::move(c._imag)) {}
    

    constexpr T real() const { return _real; }
    T& real() { return _real; }    
    constexpr T imag() const { return _imag; }
    T& imag() { return _imag; }

    
    __attribute__((always_inline)) inline Complex &  operator=(const Complex &other) {
        _real = other._real;
        _imag = other._imag;
        return *this;
    }

    __attribute__((always_inline)) inline Complex &  operator+=(const Complex &other) {
        _real += other._real;
        _imag += other._imag;
        return *this;
    }

    __attribute__((always_inline)) inline Complex &  operator-=(const Complex &other) {
        _real -= other._real;
        _imag -= other._imag;
        return *this;
    }

    __attribute__((always_inline)) inline Complex  operator+(const Complex &other) const {
        return Complex(_real + other._real, _imag + other._imag);
    }

    __attribute__((always_inline)) inline Complex  operator-(const Complex &other) const {
        return Complex(_real - other._real, _imag - other._imag);
    }

    __attribute__((always_inline)) inline Complex  operator*(const Complex &other) const {
        return Complex(_real * other._real - _imag * other._imag, _real * other._imag + _imag * other._real);
    }

    __attribute__((always_inline)) inline Complex  operator*(const T &scalar) const {
        return Complex(_real * scalar, _imag * scalar);
    }

    __attribute__((always_inline)) inline Complex  operator/(const T &scalar) const {
        return Complex(_real / scalar, _imag / scalar);
    }

    __attribute__((always_inline)) inline Complex  operator+(const T &scalar) const {
        return Complex(_real + scalar, _imag);
    }

    __attribute__((always_inline)) inline Complex  operator-(const T &scalar) const {
        return Complex(_real - scalar, _imag);
    }

    __attribute__((always_inline)) inline T  Mul_only_Real_result(const Complex &other) const {
        return _real * other._real - _imag * other._imag;
    }
};