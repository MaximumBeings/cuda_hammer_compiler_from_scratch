#define main chapter17_example_main
#include "../docs/part4/code/041_lowering_a_whole_fused_graph.cpp"
#undef main
#include <cstdint>
#include <cstring>
#include <locale>
struct CommaPunctuation : std::numpunct<char> { char do_decimal_point() const override {return ',';} };
int main() {
    auto writeFile=[](const std::string& path,const std::string& contents){std::ofstream f(path);f<<contents;f.close();if(!f)throw std::runtime_error("test source write failed");};
    std::vector<float> samples={0.0f,-0.0f,1e-7f,std::nextafter(1.0f,2.0f),0.1f,
        std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),std::numeric_limits<float>::lowest()};
    std::locale previous=std::locale(); std::locale::global(std::locale(previous,new CommaPunctuation));
    std::string source="#include <cstdint>\n#include <cstring>\n#include <cstdio>\nint main(){\n";
    for(size_t i=0;i<samples.size();++i){
        std::uint32_t bits;std::memcpy(&bits,&samples[i],sizeof bits);
        source+="{ float value="+floatLiteral(samples[i])+"; std::uint32_t bits; std::memcpy(&bits,&value,sizeof bits); if(bits!="+std::to_string(bits)+"u)return 1; }\n";
    }
    std::locale::global(previous);
    source+="std::puts(\"all9 finite literal bit patterns preserved, including negative zero and subnormal\");}\n";
    writeFile("/tmp/hammer_float_roundtrip.cpp",source);
    runShellCaptureStdout("g++ -std=c++17 -O2 /tmp/hammer_float_roundtrip.cpp -o /tmp/hammer_float_roundtrip 2>&1");
    std::printf("%s",runShellCaptureStdout("/tmp/hammer_float_roundtrip").c_str());
    bool rejects=false;try{floatLiteral(std::numeric_limits<float>::infinity());}catch(const std::invalid_argument&){rejects=true;}if(!rejects)return 2;
    bool badTokenRejected=false;try{parseFiniteFloat("1e-999");}catch(const std::runtime_error&){badTokenRejected=true;}if(!badTokenRejected)return 6;
    Graph g;auto a=g.addInput("a");auto c=g.addConst(1e-7f,"c");g.addBinary(OpKind::Mul,a,c,"out");
    auto shapes=inferShapes(g,{{a.nodeId,Shape{{1}}},{c.nodeId,Shape{}}});std::map<int,long long> counts;for(auto& kv:shapes)counts[kv.first]=numElements(kv.second);
    std::map<std::string,std::vector<float>> inputs={{"a",{1e8f}}};auto expected=evaluateArrays(g,inputs,counts);
    writeFile("/tmp/hammer_float_graph.cpp",generateFullProgram(g,shapes,counts,inputs));
    runShellCaptureStdout("g++ -std=c++17 -O2 /tmp/hammer_float_graph.cpp -o /tmp/hammer_float_graph 2>&1");
    auto actual=parseNamedBuffers(runShellCaptureStdout("/tmp/hammer_float_graph"));
    if(actual.at("out")[0]!=expected.at("out")[0] || actual.at("out")[0]!=10.0f)return 3;
    if(actual.at("c")[0]!=1e-7f)return 4;
    Graph identity;auto x=identity.addInput("x");auto one=identity.addConst(1.0f,"one");identity.addBinary(OpKind::Mul,x,one,"out");
    auto identityShapes=inferShapes(identity,{{x.nodeId,Shape{{static_cast<int>(samples.size())}}},{one.nodeId,Shape{}}});
    std::map<int,long long> identityCounts;for(auto& kv:identityShapes)identityCounts[kv.first]=numElements(kv.second);
    writeFile("/tmp/hammer_float_graph.cpp",generateFullProgram(identity,identityShapes,identityCounts,{{"x",samples}}));
    runShellCaptureStdout("g++ -std=c++17 -O2 /tmp/hammer_float_graph.cpp -o /tmp/hammer_float_graph 2>&1");
    auto identityActual=parseNamedBuffers(runShellCaptureStdout("/tmp/hammer_float_graph"));
    for(size_t i=0;i<samples.size();++i){std::uint32_t wanted,got;std::memcpy(&wanted,&samples[i],4);std::memcpy(&got,&identityActual.at("x")[i],4);if(wanted!=got){std::fprintf(stderr,"sample%zu wanted%08x got%08x\n",i,wanted,got);return 5;}}
    std::puts("all9 finite float comparison stdout values preserve their bits");
    std::puts("generated 1e8 * 1e-7 =10; source literals and comparison stdout both round-trip");
}
