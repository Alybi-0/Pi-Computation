#include <iostream>
#include <future>
#include <random>
#include <sstream>
#include <chrono>
#include <raylib.h>
#include <omp.h>
#include "coms.h"
#include "mmath.h"
#include "rain.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const uint8_t used_cores = 2U;

u_lli drops(const u_lli& RAIN, const u_li& R, uint64_t* seeds);
d_li calcPI(u_lli A, u_lli C);
const Vector2 SetDisplay(std::vector<Drop>& rain, int* data, size_t vecSize);
void DisplayDrops(std::vector<Drop>& rain, const Vector2 squarePos, int* data, size_t vecSize);

int main(int argc, char** argv)
{
    u_lli RAIN = 2000000000;
    u_li R = 1000000000;
    if(argc > 1)
    {
        std::stringstream(argv[1]) >> RAIN;
    }
    if(argc > 2)
    {
        std::stringstream(argv[2]) >> R;
    }

    size_t d_rain = 40000;
    std::vector<Drop> rainVec;
    int data[] = {1400, 900, 700};
    const Vector2 squarePos = SetDisplay(rainVec, data, d_rain);

    std::future<void> disD = std::async(std::launch::async, DisplayDrops, std::ref(rainVec), squarePos, data, d_rain);

    int thrds = omp_get_max_threads();
    uint64_t* seeds = allocate<uint64_t>(thrds);
    std::mt19937_64 seeder(std::random_device{}());
    for(int l = 0; l < thrds; l++)
    {
        seeds[l] = seeder() ^ (0x9e3779b97f4a7c15ULL + (uint64_t)l * 0x9e3779b97f4a7c15ULL);
    }

    omp_set_nested(1);
    omp_set_num_threads(omp_get_max_threads()/used_cores - 1);

    std::future<u_lli>* Drps = allocate<std::future<u_lli>>(used_cores);
    auto start = std::chrono::high_resolution_clock::now();

    for(uint8_t i = 0; i < used_cores; i++)
    {
        Drps[i] = std::async(std::launch::async, drops, RAIN/used_cores, R, seeds);
    }
    u_lli C = 0LLU;
    for(uint8_t l = 0; l < used_cores; l++)
    {
        C += Drps[l].get();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end - start;

    d_li C_PI = calcPI(RAIN, C);
    double err = (1-(M_PI/C_PI))*100.0;
    printf("Time: %.5f, RAIN: %llu, R: %lu, Pi is: %.12Lf, Error: %.8f%%\n", time.count(), RAIN, R, C_PI, std::fabs(err));
    
    delete[] Drps;
    delete[] seeds;
    std::cin.get();
    std::cout << "\033c" << std::flush;
}

u_lli drops(const u_lli& RAIN, const u_li& R, uint64_t* seeds)
{
    u_lli dC = 0;

    uint64_t outer_core = std::hash<std::thread::id>{}(std::this_thread::get_id());
    #pragma omp parallel reduction(+:dC)
    {
        int thrd_i = omp_get_thread_num();
        std::mt19937_64 nR(seeds[thrd_i]*outer_core + thrd_i);
        std::uniform_int_distribution<u_li> dist(0, R);
        u_li x, y;
        d_li r;
        
        #pragma omp for simd schedule(static)
        for(u_lli i = 0; i < RAIN; i++)
        {
            x = dist(nR);
            y = dist(nR);
            r = hypotenus(x, y);
            if(r < (d_li)R) dC++;
        }
    }

    return dC;
}

d_li calcPI(u_lli A, u_lli C)
{
    d_li Areas_r = ((d_li)C)/((d_li)A);
    return 4.0*Areas_r;
}

const Vector2 SetDisplay(std::vector<Drop>& rain, int* data, size_t vecSize)
{

    const Vector2 squarePos = {(data[0] - data[2]) / 2.0f, (data[1] - data[2]) / 2.0f};
    Drop::maxY = squarePos.y + data[2] - 3;
    Drop::minY = squarePos.y + 3;
    Drop::maxX = squarePos.x + data[2] - 2;
    Drop::minX = squarePos.x + 2;

    rain.reserve(vecSize);
    for(size_t i = 0; i < vecSize; i++)
    {
        rain.emplace_back(Drop());
    }
    return squarePos;
}

void DisplayDrops(std::vector<Drop>& rain, const Vector2 squarePos, int* data, size_t vecSize)
{
    SetTraceLogLevel(LOG_NONE);
    InitWindow(data[0], data[1], "MonteCarlo Method");

    const float border = 3.0f;

    const Vector2 cO = {squarePos.x + data[2] / 2.0f, squarePos.y + data[2] / 2.0f};
    const float cR = data[2] / 2.0f;


    uint16_t setFps = 60;
    SetTargetFPS(setFps);
    double dTime = 1.0/((double)setFps);

    BeginDrawing();
    ClearBackground(RAYWHITE);
    EndDrawing();

    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        DrawRectangleLinesEx((Rectangle){squarePos.x, squarePos.y, (float)data[2], (float)data[2]}, (int)border, BLACK);

        DrawCircleV(cO, cR+(int)border - 1, BLACK);
        DrawCircleV(cO, cR, WHITE);

        DrawText(TextFormat("Points in Circle: %i/%i", Drop::PIn, vecSize), 40, 70, 20, BLACK);

        for(Drop& d : rain)
        {
            d.upD(dTime);
            if(d.Y() <= 0)
                continue;
            d.draw();
        }
        EndDrawing();
    }

    printf("%.8f\n", dTime);
    CloseWindow();
    //std::cin.get();
    //std::cout << "\033c" << std::flush;
}
