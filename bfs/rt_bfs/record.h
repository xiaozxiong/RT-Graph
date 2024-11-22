#pragma once

#include <optix.h>

template <typename T> struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData {};
typedef Record<EmptyData> EmptyRecord;
/*
No data needed
struct RayGenData{};
struct MissData{};
struct HitGroupData{};
*/

typedef EmptyRecord RayGenRecord;
typedef EmptyRecord MissRecord;
typedef EmptyRecord HitGroupRecord;
