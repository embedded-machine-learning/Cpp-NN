#pragma once

#include <tuple>
#include <type_traits>
#include <algorithm>

template <typename T>
concept tuple_like = requires {
    typename std::tuple_size<std::remove_cvref_t<T>>::type;
};

template<typename T, typename... Ts>
consteval std::common_type_t<T, Ts...> vmax(const T &value, const Ts &...values) {
    if constexpr (sizeof...(values) == 0) {
        return value;
    } else {
        return std::max(value, vmax(values...));
    }
}
