#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>
#include <limits>

// convolutional code related parameters
const uint8_t mem_conv = 6;
const uint8_t nstate_conv = 64;
const uint8_t n_out_conv = 2;
typedef std::array<std::array<uint8_t,2>,nstate_conv> conv_arr_t;
const uint8_t G[n_out_conv] = {0171, 0133};//octal

void generate_conv_arrays(conv_arr_t &prev_state, conv_arr_t &next_state, conv_arr_t *output);
std::vector<bool> encode(std::vector<bool> &msg, conv_arr_t&next_state, conv_arr_t *output);
std::vector<bool> read_bit_array(std::string &infile);
void write_bit_array(std::vector<bool> &outvec, std::string &outfile);
std::vector<bool> viterbi_decode(std::vector<bool> &channel_output, conv_arr_t&prev_state, conv_arr_t&next_state, conv_arr_t *output);

int main(int argc, char **argv) {
    // generate convolutional code matrices
    conv_arr_t prev_state, next_state, output[n_out_conv];
    generate_conv_arrays(prev_state, next_state, output);
    if (argc < 4) throw std::runtime_error("not enough arguments. Call as ./a.out [encode/decode] infile outfile");
    std::string mode = std::string(argv[1]);
    if (mode != "encode" && mode != "decode")
        throw std::runtime_error("invalid mode");
    std::string infile = std::string(argv[2]), outfile = std::string(argv[3]);
    if (mode == "encode") {
        std::vector<bool> msg = read_bit_array(infile);
        std::vector<bool> encoded_msg = encode(msg, next_state, output);
        write_bit_array(encoded_msg, outfile);
    }
    if (mode == "decode") {
        std::vector<bool> channel_output = read_bit_array(infile);
        std::vector<bool> decoded_msg = viterbi_decode(channel_output, prev_state, next_state, output);
        write_bit_array(decoded_msg, outfile);
    }
    return 0;
}

void generate_conv_arrays(conv_arr_t &prev_state, conv_arr_t &next_state, conv_arr_t *output) {
    for (uint8_t cur_state = 0; cur_state < nstate_conv; cur_state++) {
        next_state[cur_state][0] = (cur_state>>1);
        next_state[cur_state][1] = ((cur_state|nstate_conv)>>1);
        prev_state[cur_state][0] = (cur_state<<1)&(nstate_conv-1);
        prev_state[cur_state][1] = ((cur_state<<1)|1)&(nstate_conv-1);
        output[0][cur_state][0] = __builtin_parity((cur_state&G[0]));
        output[0][cur_state][1] = __builtin_parity(((cur_state|nstate_conv)&G[0]));
        output[1][cur_state][0] = __builtin_parity((cur_state&G[1]));
        output[1][cur_state][1] = __builtin_parity(((cur_state|nstate_conv)&G[1]));
    } 
    return;
}

std::vector<bool> encode(std::vector<bool> &msg, conv_arr_t&next_state, conv_arr_t *output) {
    std::vector<bool> encoded_msg;
    uint8_t cur_state = 0;
    for (bool msg_bit : msg) {
        encoded_msg.push_back(output[0][cur_state][msg_bit]);
        encoded_msg.push_back(output[1][cur_state][msg_bit]);
        cur_state = next_state[cur_state][msg_bit];
    }
    // add terminating bits
    for (uint8_t i = 0; i < mem_conv; i++) { 
        encoded_msg.push_back(output[0][cur_state][0]);
        encoded_msg.push_back(output[1][cur_state][0]);
        cur_state = next_state[cur_state][0];
    }
    if (cur_state != 0)
        throw std::runtime_error("state after encoding not 0");
    return encoded_msg;
}

std::vector<bool> read_bit_array(std::string &infile) {
    std::ifstream fin(infile);
    std::vector<bool> vec;
    char ch;
    while (fin >> std::noskipws >> ch) {
        switch(ch) {
            case '0': vec.push_back(0); break;
            case '1': vec.push_back(1); break;
            default: throw std::runtime_error("invalid character in input file");          
        }
    }
    fin.close();
    return vec;
}

void write_bit_array(std::vector<bool> &outvec, std::string &outfile) {
    std::ofstream fout(outfile);
    for(bool b : outvec) 
        fout << (b?'1':'0');
    fout.close();
}

std::vector<bool> viterbi_decode(std::vector<bool> &channel_output, conv_arr_t&prev_state, conv_arr_t&next_state, conv_arr_t *output) {
    double INF = std::numeric_limits<double>::infinity();
    uint32_t out_size = channel_output.size();
    if (out_size%n_out_conv != 0) throw std::runtime_error("length not multiple of n_out_conv");
    uint32_t in_size = out_size/n_out_conv;
    if (in_size < (uint32_t)mem_conv) throw std::runtime_error("too small channel output");
    std::vector<std::array<uint8_t,nstate_conv>> traceback(in_size);
    std::array<double,nstate_conv> curr_score, prev_score;
    curr_score[0] = 0.0;
    for (uint8_t init_state = 1; init_state < nstate_conv; init_state++)
        curr_score[init_state] = -INF; // initial state is 0, so rest have score -inf
    for (uint32_t t = 0; t < in_size; t++) {
        prev_score = curr_score;
        for (uint8_t st2 = 0; st2 < nstate_conv; st2++) {
            // st2 = next state
            uint8_t st1 = prev_state[st2][0];
            bool curr_bit = (st2>>(mem_conv-1));
            curr_score[st2] = prev_score[st1] - (double)(channel_output[2*t]!=output[0][st1][curr_bit]) - (double)(channel_output[2*t+1]!=output[1][st1][curr_bit]);
            traceback[t][st2] = st1;
            st1 = prev_state[st2][1];
            double score = prev_score[st1] - (double)(channel_output[2*t]!=output[0][st1][curr_bit]) - (double)(channel_output[2*t+1]!=output[1][st1][curr_bit]);
            if (score > curr_score[st2]) {
                curr_score[st2] = score;
                traceback[t][st2] = st1;
            }
        }
    }
    std::vector<bool> decoded_msg(in_size);
    uint8_t cur_state = 0; // we already know the last state is 0
    decoded_msg[in_size-1] = (cur_state>>(mem_conv-1));
    for (uint32_t t = in_size-1; t > 0; t--) {
        cur_state = traceback[t][cur_state];
        decoded_msg[t-1] = (cur_state>>(mem_conv-1));
    }
    decoded_msg.resize(in_size-mem_conv);
    return decoded_msg;
}
