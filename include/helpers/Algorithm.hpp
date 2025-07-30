#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

namespace helpers {

auto max_lambda = [](auto a, auto b) { return (a > b) ? a : b; };
auto sum_lambda = [](auto a, auto b) { return a + b; };

template <typename Lambda, typename Indexes, typename... T>
struct max_helper;

template <typename Lambda>
struct max_helper<Lambda, std::index_sequence<>> {
    static constexpr void value(const std::tuple<> &tuple, const Lambda &lambda) {
    }
};

template <typename Lambda, size_t Index, typename... T>
struct max_helper<Lambda, std::index_sequence<Index>, T...> {
    static constexpr auto value(const std::tuple<T...> &tuple, const Lambda &lambda) {
        return std::get<Index>(tuple);
    }
};

template <typename Lambda, size_t Index, size_t... Indexs, typename... T>
struct max_helper<Lambda, std::index_sequence<Index, Indexs...>, T...> {
    static constexpr auto value(const std::tuple<T...> &tuple, const Lambda &lambda) {
        auto a = std::get<Index>(tuple);
        auto b = max_helper<Lambda, std::index_sequence<Indexs...>, T...>::value(tuple, lambda);
        return lambda(a, b);
    }
};

template <typename... T>
constexpr auto max(const std::tuple<T...> &tuple) {
    return max_helper<decltype(max_lambda), std::make_index_sequence<sizeof...(T)>, T...>::value(tuple, max_lambda);
}

template <typename... T>
constexpr auto max(const T&...  values) {
    return max_helper<decltype(max_lambda), std::make_index_sequence<sizeof...(T)>, T...>::value(std::tuple<T...>(values...), max_lambda);
}

template <typename... T>
constexpr auto sum(const std::tuple<T...> &tuple) {
    return max_helper<decltype(sum_lambda), std::index_sequence_for<T...>, T...>::value(tuple, sum_lambda);
}

template< typename T>
constexpr T highest_restless_division_factor_up_to(T value, T upto ){
    for( T i = upto; i > 1; i-- ){
        if( value % i == 0 ){
            return i;
        }
    }
    return 1;
}

template <size_t CharLength>
constexpr bool char_Array_contains(const std::array<char, CharLength>& str, const char c) {
    bool ret=false;
    for (size_t i = 0; i < CharLength; i++) {
        if (str[i] == c) {
            ret = true;
            break;
        }
    }
    return ret;
}

}; // namespace helpers