#define GCC_VERSION (__GNUC__ * 10000 \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)

#if GCC_VERSION >= 70000
#define if_constexpr(expression) if constexpr (expression)
#else
#define if_constexpr(expression) if(expression)
#endif
