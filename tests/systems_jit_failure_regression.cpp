// The preceding library must never be treated as the output of a failed build.
#define main chapter23_example_main
#include "../docs/part5/code/057_real_codegen_for_a_scheduled_loop_tile_size_and_unroll_factor.cpp"
#undef main
#include <filesystem>
#include <cstdlib>
int main() {
    char directory[] = "/tmp/hammer-jit-regression-XXXXXX";
    if (!mkdtemp(directory)) return 1;
    std::filesystem::path dir(directory);
    std::string stem = (dir / "source with ' quote").string();
    std::string log;
    auto loadValue = [&]() { JitModule module(stem + ".so"); return module.getFunction<int(*)()>("value")(); };
    auto mustReject = [&](const std::string& source) {
        try { compileToSharedLibrary(source, stem + ".cpp", stem + ".so", log); return false; }
        catch (const std::runtime_error&) { return true; }
    };
    try {
        if (!compileToSharedLibrary("extern \"C\" int value(){return 37;}",stem+".cpp",stem+".so",log) || loadValue()!=37) return 2;
        if (!mustReject("invalid source")) return 3;
        std::puts("diagnostic compile failure rejected before load/timing; previous artifact remains intact");
        if (loadValue()!=37) return 4;
        std::string oldPath=std::getenv("PATH") ? std::getenv("PATH") : "";
        auto fake = dir / "g++";
        writeFile(fake.string(),"#!/bin/sh\nexit 1\n");
        std::filesystem::permissions(fake,std::filesystem::perms::owner_all);
        std::string path=dir.string()+":"+oldPath; setenv("PATH",path.c_str(),1);
        bool rejected=mustReject("extern \"C\" int value(){return 99;}");
        setenv("PATH",oldPath.c_str(),1);
        if (!rejected || loadValue()!=37) return 5;
        std::puts("silent compiler exit1 rejected; stale function was not accepted as a new candidate");
        if (!compileToSharedLibrary("#warning diagnostic_is_not_failure\nextern \"C\" int value(){return 99;}",stem+".cpp",stem+".so",log) || loadValue()!=99 || log.empty()) return 6;
        std::puts("successful compiler warning retained; new staged artifact promoted and returns99");
        for (const auto& entry:std::filesystem::directory_iterator(dir))
            if (entry.path().filename().string().find(".candidate.")!=std::string::npos) return 7;
        std::filesystem::remove_all(dir);
        std::puts("unique staged source/output files cleaned after success and failure");
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 8; }
}
