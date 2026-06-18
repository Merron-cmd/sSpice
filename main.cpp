#include <bits/stdc++.h>
#include <cmath>
using namespace std;

// ==================== Status ====================
bool nonlinear;

// ==================== Constants ====================
const double PI = acos(-1.0);

// ==================== Complex Number Struct & Operations ====================
struct node {
    double volt_re, volt_im;
    node(double r = 0.0, double i = 0.0) : volt_re(r), volt_im(i) {}
    node operator+(const node& other) const {
        return {volt_re + other.volt_re, volt_im + other.volt_im};
    }
    node operator-(const node& other) const {
        return {volt_re - other.volt_re, volt_im - other.volt_im};
    }
    node operator*(const node& other) const {
        return {volt_re * other.volt_re - volt_im * other.volt_im,
                volt_re * other.volt_im + volt_im * other.volt_re};
    }
    node operator/(const node& other) const {
        double denom = other.volt_re * other.volt_re + other.volt_im * other.volt_im;
        return {(volt_re * other.volt_re + volt_im * other.volt_im) / denom,
                (volt_im * other.volt_re - volt_re * other.volt_im) / denom};
    }
};

#define GMIN 1e-12
#define VNTOL 1e-6
#define ABSTOL 1e-12
#define RELTOL 1e-4

// ==================== Function Declarations ====================
double norm2(const node& z);
void matrix_mul_vec(int n, node** A, node* x, node* ans);
void gauss_elimination(int n, node** A, node* b, node* x);
void read_netlist(istream& in);
void build_dc_mna(int n_nodes, int n_vs, node** A, node* b);
void build_dc_mna_nonlinear(int n_nodes, int n_vs, node** A, node* b, node* x_last);
void build_ac_mna(int n_nodes, int n_vs, double freq, const string& keep_name,
                  double amp, node** A, node* b);
string get_basename(const string& filename);
vector<double> generate_freq_points(double start, double end, int points, int type);
void solve_dc_operating_point(ostream& out = cout, bool print_output = true);
void solve_ac_single_frequency(double freq, ostream& out = cout, bool with_header = true);
void solve_ac_sweep(double start, double end, int points, int sweep_type,
                    const string& keep_name, double amplitude, ostream& out = cout);

string format_double(double val, int precision = 8);

// Global variables
string current_filename;
int c_size;
enum SourceType { DC_SOURCE, AC_SOURCE };

struct VoltageSource {
    string name; int node1, node2; SourceType type;
    double dc_value, ac_magnitude, ac_freq, ac_phase;
};
struct CurrentSource {
    string name; int node1, node2; SourceType type;
    double dc_value, ac_magnitude, ac_freq, ac_phase;
};
struct Resistor { string name; int node1, node2; double value; };
struct Capacitor { string name; int node1, node2; double value; };
struct Inductor { string name; int node1, node2; double value; };

const double Is = 1e-14;
const double VT = 0.026;

struct Diode {
    string name;
    int node1, node2;
};

struct VCVS {
    string name;
    int node1, node2;
    int node3, node4;
    double gain;
};

struct VCCS {
    string name;
    int node1, node2;
    int node3, node4;
    double trans;
};

struct CCCS {
    string name;
    int node1, node2;
    string ctrl_vsource_name;   // 控制电压源的名字
    double gain;
};

struct CCVS {
    string name;
    int node1, node2;
    string ctrl_vsource_name;   // 控制电压源的名字
    double trans;
};

vector<VoltageSource> voltage_sources;
vector<CurrentSource> current_sources;
vector<Resistor> resistors;
vector<Capacitor> capacitors;
vector<Inductor> inductors;
vector<Diode> diodes;
vector<VCVS> vcvs;
vector<VCCS> vccs;
vector<CCCS> cccs;
vector<CCVS> ccvs;

vector<double> dc_voltages;  // DC node voltages

// ==================== Utility Functions ====================
double norm2(const node& z) {
    return z.volt_re * z.volt_re + z.volt_im * z.volt_im;
}

void matrix_mul_vec(int n, node** A, node* x, node* ans) {
    for(int i = 0; i < n; ++i) {
        ans[i] = node(0, 0);
        for(int j = 0; j < n; ++j) {
            ans[i] = ans[i] + A[i][j] * x[j];
        }
    }
}

void gauss_elimination(int n, node** A, node* b, node* x) {
    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double maxNorm = norm2(A[k][k]);
        for (int i = k + 1; i < n; ++i) {
            double cur = norm2(A[i][k]);
            if (cur > maxNorm) { maxNorm = cur; pivot = i; }
        }
        if (maxNorm < 1e-15) {
            cerr << "Error: Singular matrix!" << endl;
            return;
        }
        if (pivot != k) {
            swap(A[k], A[pivot]);
            swap(b[k], b[pivot]);
        }
        node pivotVal = A[k][k];
        for (int i = k + 1; i < n; ++i) {
            node factor = A[i][k] / pivotVal;
            for (int j = k; j < n; ++j)
                A[i][j] = A[i][j] - factor * A[k][j];
            b[i] = b[i] - factor * b[k];
            A[i][k] = {0.0, 0.0};
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        node sum = b[i];
        for (int j = i + 1; j < n; ++j)
            sum = sum - A[i][j] * x[j];
        x[i] = sum / A[i][i];
    }
}

string format_double(double val, int precision) {
    ostringstream oss;
    if (fabs(val) < 1e-3 && val != 0.0) {
        oss << scientific << setprecision(precision) << val;
    } else {
        oss << fixed << setprecision(precision) << val;
    }
    return oss.str();
}

double parse_value(const string& s) {
    if (s.empty()) return 0.0;
    static const unordered_map<char, double> prefix = {
        {'u', 1e-6}, {'U', 1e-6},
        {'m', 1e-3},
        {'k', 1e3}, {'K', 1e3},
        {'M', 1e6},
        {'G', 1e9},
        {'T', 1e12},
        {'n', 1e-9}, {'N', 1e-9},
        {'p', 1e-12}, {'P', 1e-12},
        {'f', 1e-15}, {'F', 1e-15},
        {'a', 1e-18}, {'A', 1e-18}
    };
    char last = s.back();
    double scale = 1.0;
    string num_part = s;
    if (isalpha(last)) {
        auto it = prefix.find(last);
        if (it != prefix.end()) {
            scale = it->second;
            num_part = s.substr(0, s.size() - 1);
        } else {
            cerr << "Warning: unknown prefix '" << last << "' ignored." << endl;
        }
    }
    try { return stod(num_part) * scale; }
    catch (...) { cerr << "Error: invalid numeric format '" << s << "'" << endl; return 0.0; }
}

double read_double_with_units(istream& in) {
    string s;
    in >> s;
    return parse_value(s);
}

int read_int(const string& prompt) {
    int value;
    while (true) {
        cout << prompt;
        if (cin >> value) {
            break;
        } else {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input. Please enter a number." << endl;
        }
    }
    return value;
}

// ==================== Netlist Reader ====================
void read_netlist(istream& in) {
    in >> c_size;
    bool islinear = 1;
    for (int i = 0; i < c_size; ++i) {
        string name; int n1, n2; string token;
        in >> name >> n1 >> n2 >> token;
        char type = name[0];
        switch (type) {
            case 'V': case 'v': {
                VoltageSource vs{name, n1, n2, DC_SOURCE, 0, 0, 0, 0};
                if (token == "DC" || token == "dc") {
                    string val; in >> val;
                    vs.type = DC_SOURCE; vs.dc_value = parse_value(val);
                } else if (token == "AC" || token == "ac") {
                    string mag, freq, ph; in >> mag >> freq >> ph;
                    vs.type = AC_SOURCE;
                    vs.ac_magnitude = parse_value(mag);
                    vs.ac_freq = parse_value(freq);
                    vs.ac_phase = parse_value(ph);
                    vs.dc_value = 0.0;
                } else {
                    vs.type = DC_SOURCE;
                    vs.dc_value = parse_value(token);
                }
                voltage_sources.push_back(vs);
                break;
            }
            case 'I': case 'i': {
                CurrentSource cs{name, n1, n2, DC_SOURCE, 0, 0, 0, 0};
                if (token == "DC" || token == "dc") {
                    string val; in >> val;
                    cs.type = DC_SOURCE; cs.dc_value = parse_value(val);
                } else if (token == "AC" || token == "ac") {
                    string mag, freq, ph; in >> mag >> freq >> ph;
                    cs.type = AC_SOURCE;
                    cs.ac_magnitude = parse_value(mag);
                    cs.ac_freq = parse_value(freq);
                    cs.ac_phase = parse_value(ph);
                    cs.dc_value = 0.0;
                } else {
                    cs.type = DC_SOURCE;
                    cs.dc_value = parse_value(token);
                }
                current_sources.push_back(cs);
                break;
            }
            case 'R': case 'r':
                resistors.push_back({name, n1, n2, parse_value(token)});
                break;
            case 'C': case 'c':
                capacitors.push_back({name, n1, n2, parse_value(token)});
                break;
            case 'L': case 'l':
                inductors.push_back({name, n1, n2, parse_value(token)});
                break;
            case 'D': case 'd':
                islinear = 0;
                diodes.push_back({name, n1, n2});
                break;
            case 'E': case 'e': {
                int n3, n4;
                double gain;
                n3 = stoi(token);
                in >> n4 >> gain;
                vcvs.push_back({name, n1, n2, n3, n4, gain});
                break;
            }
            case 'G': case 'g': {
                int n3, n4;
                double trans;
                n3 = stoi(token);
                in >> n4 >> trans;
                vccs.push_back({name, n1, n2, n3, n4, trans});
                break;
            }
            case 'F': case 'f': {
                string ctrl_name;
                double gain;
                ctrl_name = token;
                in >> gain;
                cccs.push_back({name, n1, n2, ctrl_name, gain});
                break;
            }
            case 'H': case 'h': {
                string ctrl_name;
                double trans;
                ctrl_name = token;
                in >> trans;
                ccvs.push_back({name, n1, n2, ctrl_name, trans});
                break;
            }
            default:
                cerr << "Error: unknown component type '" << type << "'" << endl;
        }
    }
    if(islinear == 1) nonlinear = 0;
    else nonlinear = 1;
    // Check ground
    bool has_gnd = false;
    for (auto& v : voltage_sources) if (v.node1==0 || v.node2==0) has_gnd = true;
    for (auto& i : current_sources) if (i.node1==0 || i.node2==0) has_gnd = true;
    for (auto& r : resistors)       if (r.node1==0 || r.node2==0) has_gnd = true;
    for (auto& c : capacitors)      if (c.node1==0 || c.node2==0) has_gnd = true;
    for (auto& l : inductors)       if (l.node1==0 || l.node2==0) has_gnd = true;
    for (auto& e : vcvs)            if (e.node1==0 || e.node2==0 || e.node3==0 || e.node4==0) has_gnd = true;
    for (auto& g : vccs)            if (g.node1==0 || g.node2==0 || g.node3==0 || g.node4==0) has_gnd = true;
    for (auto& f : cccs)            if (f.node1==0 || f.node2==0) has_gnd = true;
    for (auto& h : ccvs)            if (h.node1==0 || h.node2==0) has_gnd = true;
    if (!has_gnd) cerr << "Warning: No component connected to GND (node 0). Circuit may be floating!" << endl;
}

string get_basename(const string& filename) {
    size_t dot = filename.find_last_of('.');
    if (dot != string::npos) return filename.substr(0, dot);
    return filename;
}

vector<double> generate_freq_points(double start, double end, int points, int type) {
    vector<double> freqs;
    if (type == 1) {
        if (points <= 1) { freqs.push_back(start); return freqs; }
        double step = (end - start) / (points - 1);
        for (int i = 0; i < points; ++i)
            freqs.push_back(start + i * step);
    } else {
        if (start <= 0 || end <= 0 || points <= 0) {
            cerr << "Error: start, end must be positive and points > 0 for log sweep." << endl;
            return freqs;
        }
        double ratio = pow(10.0, 1.0 / points);
        double f = start;
        const double eps = 1e-12;
        while (f <= end * (1.0 + eps)) {
            freqs.push_back(f);
            f *= ratio;
        }
        if (freqs.back() < end - eps) {
            freqs.push_back(end);
        } else if (fabs(freqs.back() - end) > eps) {
            freqs.back() = end;
        }
    }
    return freqs;
}

// ==================== MNA Build Functions ====================
// Helper to create name->index map for current variables
static unordered_map<string, int> build_name_index_map(int n_nodes, int &n_vs) {
    unordered_map<string, int> mp;
    int vs_idx = n_nodes;
    for (auto& v : voltage_sources) mp[v.name] = vs_idx++;
    for (auto& l : inductors)       mp[l.name] = vs_idx++;
    for (auto& e : vcvs)            mp[e.name] = vs_idx++;
    for (auto& h : ccvs)            mp[h.name] = vs_idx++;
    n_vs = vs_idx - n_nodes;
    return mp;
}

// DC linear MNA
void build_dc_mna(int n_nodes, int n_vs, node** A, node* b) {
    auto idx = [&](int node) { return (node == 0) ? -1 : node - 1; };
    // Resistors
    for (auto& r : resistors) {
        double g = 1.0 / r.value;
        int p = idx(r.node1), q = idx(r.node2);
        if (p != -1) { A[p][p] = A[p][p] + node(g,0); if(q!=-1) A[p][q] = A[p][q] - node(g,0); }
        if (q != -1) { A[q][q] = A[q][q] + node(g,0); if(p!=-1) A[q][p] = A[q][p] - node(g,0); }
    }
    // Current sources
    for (auto& i : current_sources) {
        double I = i.dc_value;
        int p = idx(i.node1), q = idx(i.node2);
        if (p != -1) b[p] = b[p] + node(I,0);
        if (q != -1) b[q] = b[q] - node(I,0);
    }

    int vs_idx = n_nodes;
    unordered_map<string, int> name_to_idx = build_name_index_map(n_nodes, n_vs); // n_vs is passed by reference

    // Independent voltage sources
    for (auto& v : voltage_sources) {
        int p = idx(v.node1), q = idx(v.node2);
        double V = v.dc_value;
        if (p != -1) A[p][vs_idx] = A[p][vs_idx] + node(1,0);
        if (q != -1) A[q][vs_idx] = A[q][vs_idx] - node(1,0);
        if (p != -1) A[vs_idx][p] = A[vs_idx][p] + node(1,0);
        if (q != -1) A[vs_idx][q] = A[vs_idx][q] - node(1,0);
        b[vs_idx] = node(V,0);
        vs_idx++;
    }

    // Inductors (short circuit in DC)
    for (auto& l : inductors) {
        int p = idx(l.node1), q = idx(l.node2);
        if (p != -1) A[p][vs_idx] = A[p][vs_idx] + node(1,0);
        if (q != -1) A[q][vs_idx] = A[q][vs_idx] - node(1,0);
        if (p != -1) A[vs_idx][p] = A[vs_idx][p] + node(1,0);
        if (q != -1) A[vs_idx][q] = A[vs_idx][q] - node(1,0);
        b[vs_idx] = node(0,0);
        vs_idx++;
    }

    // VCVS (E)
    for (auto& e : vcvs) {
        int p_out = idx(e.node1), q_out = idx(e.node2);
        int p_ctrl = idx(e.node3), q_ctrl = idx(e.node4);
        double gain = e.gain;
        if (p_out != -1) A[p_out][vs_idx] = A[p_out][vs_idx] + node(1,0);
        if (q_out != -1) A[q_out][vs_idx] = A[q_out][vs_idx] - node(1,0);
        if (p_out != -1) A[vs_idx][p_out] = A[vs_idx][p_out] + node(1,0);
        if (q_out != -1) A[vs_idx][q_out] = A[vs_idx][q_out] - node(1,0);
        if (p_ctrl != -1) A[vs_idx][p_ctrl] = A[vs_idx][p_ctrl] - node(gain,0);
        if (q_ctrl != -1) A[vs_idx][q_ctrl] = A[vs_idx][q_ctrl] + node(gain,0);
        vs_idx++;
    }

    // CCVS (H)
    for (auto& h : ccvs) {
        int p_out = idx(h.node1), q_out = idx(h.node2);
        double trans = h.trans;
        auto it = name_to_idx.find(h.ctrl_vsource_name);
        if (it == name_to_idx.end()) {
            cerr << "Error: CCVS " << h.name << " controls unknown source " << h.ctrl_vsource_name << endl;
            continue;
        }
        int ctrl_idx = it->second;
        if (p_out != -1) A[p_out][vs_idx] = A[p_out][vs_idx] + node(1,0);
        if (q_out != -1) A[q_out][vs_idx] = A[q_out][vs_idx] - node(1,0);
        if (p_out != -1) A[vs_idx][p_out] = A[vs_idx][p_out] + node(1,0);
        if (q_out != -1) A[vs_idx][q_out] = A[vs_idx][q_out] - node(1,0);
        A[vs_idx][ctrl_idx] = A[vs_idx][ctrl_idx] + node(trans,0);
        vs_idx++;
    }

    // VCCS (G)
    for (auto& g : vccs) {
        int p_out = idx(g.node1), q_out = idx(g.node2);
        int p_ctrl = idx(g.node3), q_ctrl = idx(g.node4);
        double trans = g.trans;
        if (p_out != -1) {
            if (p_ctrl != -1) A[p_out][p_ctrl] = A[p_out][p_ctrl] + node(trans,0);
            if (q_ctrl != -1) A[p_out][q_ctrl] = A[p_out][q_ctrl] - node(trans,0);
        }
        if (q_out != -1) {
            if (p_ctrl != -1) A[q_out][p_ctrl] = A[q_out][p_ctrl] - node(trans,0);
            if (q_ctrl != -1) A[q_out][q_ctrl] = A[q_out][q_ctrl] + node(trans,0);
        }
    }

    // CCCS (F)
    for (auto& f : cccs) {
        int p_out = idx(f.node1), q_out = idx(f.node2);
        double gain = f.gain;
        auto it = name_to_idx.find(f.ctrl_vsource_name);
        if (it == name_to_idx.end()) {
            cerr << "Error: CCCS " << f.name << " controls unknown source " << f.ctrl_vsource_name << endl;
            continue;
        }
        int ctrl_idx = it->second;
        if (p_out != -1) A[p_out][ctrl_idx] = A[p_out][ctrl_idx] - node(gain,0);
        if (q_out != -1) A[q_out][ctrl_idx] = A[q_out][ctrl_idx] + node(gain,0);
    }

    // Capacitors ignored in DC
}

// Nonlinear DC MNA (with diodes)
void build_dc_mna_nonlinear(int n_nodes, int n_vs, node** A, node* b, node* x_last) {
    auto idx = [&](int node) { return (node == 0) ? -1 : node - 1; };
    // Resistors
    for (auto& r : resistors) {
        double g = 1.0 / r.value;
        int p = idx(r.node1), q = idx(r.node2);
        if (p != -1) { A[p][p] = A[p][p] + node(g,0); if(q!=-1) A[p][q] = A[p][q] - node(g,0); }
        if (q != -1) { A[q][q] = A[q][q] + node(g,0); if(p!=-1) A[q][p] = A[q][p] - node(g,0); }
    }
    // Current sources
    for (auto& i : current_sources) {
        double I = i.dc_value;
        int p = idx(i.node1), q = idx(i.node2);
        if (p != -1) b[p] = b[p] + node(I,0);
        if (q != -1) b[q] = b[q] - node(I,0);
    }

    int vs_idx = n_nodes;
    unordered_map<string, int> name_to_idx;
    for (auto& v : voltage_sources) name_to_idx[v.name] = vs_idx++;
    for (auto& l : inductors)       name_to_idx[l.name] = vs_idx++;
    for (auto& e : vcvs)            name_to_idx[e.name] = vs_idx++;
    for (auto& h : ccvs)            name_to_idx[h.name] = vs_idx++;
    // n_vs is fixed, but we don't use it here

    // Independent voltage sources
    vs_idx = n_nodes;
    for (auto& v : voltage_sources) {
        int p = idx(v.node1), q = idx(v.node2);
        double V = v.dc_value;
        if (p != -1) A[p][vs_idx] = A[p][vs_idx] + node(1,0);
        if (q != -1) A[q][vs_idx] = A[q][vs_idx] - node(1,0);
        if (p != -1) A[vs_idx][p] = A[vs_idx][p] + node(1,0);
        if (q != -1) A[vs_idx][q] = A[vs_idx][q] - node(1,0);
        b[vs_idx] = node(V,0);
        vs_idx++;
    }
    // Inductors
    for (auto& l : inductors) {
        int p = idx(l.node1), q = idx(l.node2);
        if (p != -1) A[p][vs_idx] = A[p][vs_idx] + node(1,0);
        if (q != -1) A[q][vs_idx] = A[q][vs_idx] - node(1,0);
        if (p != -1) A[vs_idx][p] = A[vs_idx][p] + node(1,0);
        if (q != -1) A[vs_idx][q] = A[vs_idx][q] - node(1,0);
        b[vs_idx] = node(0,0);
        vs_idx++;
    }
    // VCVS
    for (auto& e : vcvs) {
        int p_out = idx(e.node1), q_out = idx(e.node2);
        int p_ctrl = idx(e.node3), q_ctrl = idx(e.node4);
        double gain = e.gain;
        if (p_out != -1) A[p_out][vs_idx] = A[p_out][vs_idx] + node(1,0);
        if (q_out != -1) A[q_out][vs_idx] = A[q_out][vs_idx] - node(1,0);
        if (p_out != -1) A[vs_idx][p_out] = A[vs_idx][p_out] + node(1,0);
        if (q_out != -1) A[vs_idx][q_out] = A[vs_idx][q_out] - node(1,0);
        if (p_ctrl != -1) A[vs_idx][p_ctrl] = A[vs_idx][p_ctrl] - node(gain,0);
        if (q_ctrl != -1) A[vs_idx][q_ctrl] = A[vs_idx][q_ctrl] + node(gain,0);
        vs_idx++;
    }
    // CCVS
    for (auto& h : ccvs) {
        int p_out = idx(h.node1), q_out = idx(h.node2);
        double trans = h.trans;
        auto it = name_to_idx.find(h.ctrl_vsource_name);
        if (it == name_to_idx.end()) {
            cerr << "Error: CCVS " << h.name << " controls unknown source " << h.ctrl_vsource_name << endl;
            continue;
        }
        int ctrl_idx = it->second;
        if (p_out != -1) A[p_out][vs_idx] = A[p_out][vs_idx] + node(1,0);
        if (q_out != -1) A[q_out][vs_idx] = A[q_out][vs_idx] - node(1,0);
        if (p_out != -1) A[vs_idx][p_out] = A[vs_idx][p_out] + node(1,0);
        if (q_out != -1) A[vs_idx][q_out] = A[vs_idx][q_out] - node(1,0);
        A[vs_idx][ctrl_idx] = A[vs_idx][ctrl_idx] + node(trans,0);
        vs_idx++;
    }
    // VCCS
    for (auto& g : vccs) {
        int p_out = idx(g.node1), q_out = idx(g.node2);
        int p_ctrl = idx(g.node3), q_ctrl = idx(g.node4);
        double trans = g.trans;
        if (p_out != -1) {
            if (p_ctrl != -1) A[p_out][p_ctrl] = A[p_out][p_ctrl] + node(trans,0);
            if (q_ctrl != -1) A[p_out][q_ctrl] = A[p_out][q_ctrl] - node(trans,0);
        }
        if (q_out != -1) {
            if (p_ctrl != -1) A[q_out][p_ctrl] = A[q_out][p_ctrl] - node(trans,0);
            if (q_ctrl != -1) A[q_out][q_ctrl] = A[q_out][q_ctrl] + node(trans,0);
        }
    }
    // CCCS
    for (auto& f : cccs) {
        int p_out = idx(f.node1), q_out = idx(f.node2);
        double gain = f.gain;
        auto it = name_to_idx.find(f.ctrl_vsource_name);
        if (it == name_to_idx.end()) {
            cerr << "Error: CCCS " << f.name << " controls unknown source " << f.ctrl_vsource_name << endl;
            continue;
        }
        int ctrl_idx = it->second;
        if (p_out != -1) A[p_out][ctrl_idx] = A[p_out][ctrl_idx] - node(gain,0);
        if (q_out != -1) A[q_out][ctrl_idx] = A[q_out][ctrl_idx] + node(gain,0);
    }

    // Diodes
    for (auto& d : diodes) {
        int p = idx(d.node1), q = idx(d.node2);
        double Vp = (p == -1) ? 0.0 : x_last[p].volt_re;
        double Vq = (q == -1) ? 0.0 : x_last[q].volt_re;
        double Vd = Vp - Vq;
        double g = Is / VT * exp(Vd / VT);
        double Id = Is * (exp(Vd / VT) - 1);
        double Ieq = Id - g * Vd;
        if (p != -1) {
            A[p][p] = A[p][p] + node(g,0);
            if (q != -1) A[p][q] = A[p][q] - node(g,0);
        }
        if (q != -1) {
            A[q][q] = A[q][q] + node(g,0);
            if (p != -1) A[q][p] = A[q][p] - node(g,0);
        }
        if (p != -1) b[p] = b[p] - node(Ieq,0);
        if (q != -1) b[q] = b[q] + node(Ieq,0);
    }
    // Capacitors ignored
}

// AC MNA
void build_ac_mna(int n_nodes, int n_vs, double freq, const string& keep_name,
                  double amp, node** A, node* b) {
    double omega = 2.0 * PI * freq;
    auto idx = [&](int node) { return (node == 0) ? -1 : node - 1; };

    // Resistors
    for (auto& r : resistors) {
        double g = 1.0 / r.value;
        int p = idx(r.node1), q = idx(r.node2);
        if (p != -1) { A[p][p] = A[p][p] + node(g,0); if(q!=-1) A[p][q] = A[p][q] - node(g,0); }
        if (q != -1) { A[q][q] = A[q][q] + node(g,0); if(p!=-1) A[q][p] = A[q][p] - node(g,0); }
    }
    // Capacitors Y = jωC
    for (auto& c : capacitors) {
        double y_im = omega * c.value;
        int p = idx(c.node1), q = idx(c.node2);
        if (p != -1) { A[p][p] = A[p][p] + node(0, y_im); if(q!=-1) A[p][q] = A[p][q] - node(0, y_im); }
        if (q != -1) { A[q][q] = A[q][q] + node(0, y_im); if(p!=-1) A[q][p] = A[q][p] - node(0, y_im); }
    }
    // Inductors: We will NOT use admittance, but introduce branch equation with current variable.
    // So skip admittance for inductors here.

    // Current sources (AC)
    for (auto& i : current_sources) {
        double I = 0.0;
        if (keep_name.empty()) {
            if (i.type == AC_SOURCE) {
                double phase_rad = i.ac_phase * PI / 180.0;
                I = i.ac_magnitude;
                int p = idx(i.node1), q = idx(i.node2);
                if (p != -1) b[p] = b[p] + node(I * cos(phase_rad), I * sin(phase_rad));
                if (q != -1) b[q] = b[q] - node(I * cos(phase_rad), I * sin(phase_rad));
            }
        } else {
            if (i.type == AC_SOURCE && i.name == keep_name) {
                I = amp;
                int p = idx(i.node1), q = idx(i.node2);
                if (p != -1) b[p] = b[p] + node(I, 0);
                if (q != -1) b[q] = b[q] - node(I, 0);
            }
        }
    }

    int vs_idx = n_nodes;
    unordered_map<string, int> name_to_idx;
    for (auto& v : voltage_sources) name_to_idx[v.name] = vs_idx++;
    for (auto& l : inductors)       name_to_idx[l.name] = vs_idx++;
    for (auto& e : vcvs)            name_to_idx[e.name] = vs_idx++;
    for (auto& h : ccvs)            name_to_idx[h.name] = vs_idx++;

    // Independent voltage sources
    vs_idx = n_nodes;
    for (auto& v : voltage_sources) {
        int p = idx(v.node1), q = idx(v.node2);
        double V = 0.0;
        if (keep_name.empty()) {
            if (v.type == AC_SOURCE) {
                double phase_rad = v.ac_phase * PI / 180.0;
                V = v.ac_magnitude;
                b[vs_idx] = node(V * cos(phase_rad), V * sin(phase_rad));
            }
        } else {
            if (v.type == AC_SOURCE && v.name == keep_name) {
                V = amp;
                b[vs_idx] = node(V, 0);
            }
        }
        if (p != -1) A[p][vs_idx] = A[p][vs_idx] + node(1,0);
        if (q != -1) A[q][vs_idx] = A[q][vs_idx] - node(1,0);
        if (p != -1) A[vs_idx][p] = A[vs_idx][p] + node(1,0);
        if (q != -1) A[vs_idx][q] = A[vs_idx][q] - node(1,0);
        vs_idx++;
    }

    // Inductors: branch equation V(p) - V(q) = jωL * I_L  =>  V(p)-V(q) - jωL*I = 0
    for (auto& l : inductors) {
        int p = idx(l.node1), q = idx(l.node2);
        double L = l.value;
        node impedance(0, omega * L); // jωL
        // KCL: current flows from p to q
        if (p != -1) A[p][vs_idx] = A[p][vs_idx] + node(1,0);
        if (q != -1) A[q][vs_idx] = A[q][vs_idx] - node(1,0);
        // Branch equation: V(p)-V(q) - (jωL)*I = 0
        if (p != -1) A[vs_idx][p] = A[vs_idx][p] + node(1,0);
        if (q != -1) A[vs_idx][q] = A[vs_idx][q] - node(1,0);
        A[vs_idx][vs_idx] = A[vs_idx][vs_idx] - impedance; // -jωL
        // RHS = 0
        vs_idx++;
    }

    // VCVS (E)
    for (auto& e : vcvs) {
        int p_out = idx(e.node1), q_out = idx(e.node2);
        int p_ctrl = idx(e.node3), q_ctrl = idx(e.node4);
        double gain = e.gain;
        if (p_out != -1) A[p_out][vs_idx] = A[p_out][vs_idx] + node(1,0);
        if (q_out != -1) A[q_out][vs_idx] = A[q_out][vs_idx] - node(1,0);
        if (p_out != -1) A[vs_idx][p_out] = A[vs_idx][p_out] + node(1,0);
        if (q_out != -1) A[vs_idx][q_out] = A[vs_idx][q_out] - node(1,0);
        if (p_ctrl != -1) A[vs_idx][p_ctrl] = A[vs_idx][p_ctrl] - node(gain,0);
        if (q_ctrl != -1) A[vs_idx][q_ctrl] = A[vs_idx][q_ctrl] + node(gain,0);
        vs_idx++;
    }

    // CCVS (H)
    for (auto& h : ccvs) {
        int p_out = idx(h.node1), q_out = idx(h.node2);
        double trans = h.trans;
        auto it = name_to_idx.find(h.ctrl_vsource_name);
        if (it == name_to_idx.end()) {
            cerr << "Error: CCVS " << h.name << " controls unknown source " << h.ctrl_vsource_name << endl;
            continue;
        }
        int ctrl_idx = it->second;
        if (p_out != -1) A[p_out][vs_idx] = A[p_out][vs_idx] + node(1,0);
        if (q_out != -1) A[q_out][vs_idx] = A[q_out][vs_idx] - node(1,0);
        if (p_out != -1) A[vs_idx][p_out] = A[vs_idx][p_out] + node(1,0);
        if (q_out != -1) A[vs_idx][q_out] = A[vs_idx][q_out] - node(1,0);
        A[vs_idx][ctrl_idx] = A[vs_idx][ctrl_idx] + node(trans,0);
        vs_idx++;
    }

    // VCCS (G)
    for (auto& g : vccs) {
        int p_out = idx(g.node1), q_out = idx(g.node2);
        int p_ctrl = idx(g.node3), q_ctrl = idx(g.node4);
        double trans = g.trans;
        if (p_out != -1) {
            if (p_ctrl != -1) A[p_out][p_ctrl] = A[p_out][p_ctrl] + node(trans,0);
            if (q_ctrl != -1) A[p_out][q_ctrl] = A[p_out][q_ctrl] - node(trans,0);
        }
        if (q_out != -1) {
            if (p_ctrl != -1) A[q_out][p_ctrl] = A[q_out][p_ctrl] - node(trans,0);
            if (q_ctrl != -1) A[q_out][q_ctrl] = A[q_out][q_ctrl] + node(trans,0);
        }
    }

    // CCCS (F)
    for (auto& f : cccs) {
        int p_out = idx(f.node1), q_out = idx(f.node2);
        double gain = f.gain;
        auto it = name_to_idx.find(f.ctrl_vsource_name);
        if (it == name_to_idx.end()) {
            cerr << "Error: CCCS " << f.name << " controls unknown source " << f.ctrl_vsource_name << endl;
            continue;
        }
        int ctrl_idx = it->second;
        if (p_out != -1) A[p_out][ctrl_idx] = A[p_out][ctrl_idx] - node(gain,0);
        if (q_out != -1) A[q_out][ctrl_idx] = A[q_out][ctrl_idx] + node(gain,0);
    }

    // Diodes: small-signal conductance
    for (auto& d : diodes) {
        int p = idx(d.node1), q = idx(d.node2);
        double Vp = (p == -1) ? 0.0 : dc_voltages[p];
        double Vq = (q == -1) ? 0.0 : dc_voltages[q];
        double Vd = Vp - Vq;
        double g = Is / VT * exp(Vd / VT);
        if (p != -1) {
            A[p][p] = A[p][p] + node(g,0);
            if (q != -1) A[p][q] = A[p][q] - node(g,0);
        }
        if (q != -1) {
            A[q][q] = A[q][q] + node(g,0);
            if (p != -1) A[q][p] = A[q][p] - node(g,0);
        }
    }
}

// ==================== DC Analysis ====================
void solve_dc_operating_point(ostream& out, bool print_output) {
    int max_node = 0;
    for (auto& v : voltage_sources) max_node = max(max_node, max(v.node1, v.node2));
    for (auto& i : current_sources) max_node = max(max_node, max(i.node1, i.node2));
    for (auto& r : resistors)       max_node = max(max_node, max(r.node1, r.node2));
    for (auto& c : capacitors)      max_node = max(max_node, max(c.node1, c.node2));
    for (auto& l : inductors)       max_node = max(max_node, max(l.node1, l.node2));
    for (auto& d : diodes)          max_node = max(max_node, max(d.node1, d.node2));
    for (auto& e : vcvs)            max_node = max(max_node, max(e.node1, e.node2));
    for (auto& g : vccs)            max_node = max(max_node, max(g.node1, g.node2));
    for (auto& f : cccs)            max_node = max(max_node, max(f.node1, f.node2));
    for (auto& h : ccvs)            max_node = max(max_node, max(h.node1, h.node2));

    int n_nodes = max_node;
    // Count current variables: voltage sources, inductors, VCVS, CCVS
    int n_vs = voltage_sources.size() + inductors.size() + vcvs.size() + ccvs.size();
    int total_vars = n_nodes + n_vs;
    if (total_vars == 0) {
        out << "No nodes." << endl;
        dc_voltages.clear();
        return;
    }

    node** A = new node*[total_vars];
    for (int i = 0; i < total_vars; ++i) A[i] = new node[total_vars]();
    node* b = new node[total_vars]();
    node* x = new node[total_vars]();
    for (int i = 0; i < total_vars; ++i) x[i] = node(0,0);

    if (nonlinear == 0) {
        build_dc_mna(n_nodes, n_vs, A, b);
        for (int i = 0; i < n_nodes; ++i) A[i][i] = A[i][i] + node(GMIN,0);
        gauss_elimination(total_vars, A, b, x);
    } else {
        node* x_last = new node[total_vars]();
        for (int i = 0; i < total_vars; ++i) x_last[i] = node(0,0);
        const int MAX_ITER = 1000;
        const double MAX_V_STEP = 0.1;
        const double MAX_I_STEP = 0.001;
        int iter = 0;
        bool converged = false;
        while (iter < MAX_ITER) {
            for (int i = 0; i < total_vars; ++i) {
                fill(A[i], A[i] + total_vars, node(0,0));
                b[i] = node(0,0);
            }
            build_dc_mna_nonlinear(n_nodes, n_vs, A, b, x_last);
            for (int i = 0; i < n_nodes; ++i) A[i][i] = A[i][i] + node(GMIN,0);
            gauss_elimination(total_vars, A, b, x);

            double max_delta_v = 0.0, max_delta_i = 0.0;
            for (int i = 0; i < n_nodes; ++i) {
                double delta = fabs(x[i].volt_re - x_last[i].volt_re);
                if (delta > max_delta_v) max_delta_v = delta;
            }
            for (int i = n_nodes; i < total_vars; ++i) {
                double delta = fabs(x[i].volt_re - x_last[i].volt_re);
                if (delta > max_delta_i) max_delta_i = delta;
            }
            double scale = 1.0;
            if (max_delta_v > MAX_V_STEP) scale = min(scale, MAX_V_STEP / max_delta_v);
            if (max_delta_i > MAX_I_STEP) scale = min(scale, MAX_I_STEP / max_delta_i);
            if (scale < 1.0) {
                for (int i = 0; i < total_vars; ++i) {
                    x[i].volt_re = x_last[i].volt_re + scale * (x[i].volt_re - x_last[i].volt_re);
                    x[i].volt_im = 0.0;
                }
            }

            double max_volt_change = 0.0, max_curr_change = 0.0;
            double max_volt = 0.0, max_curr = 0.0;
            double max_err = 0.0, max_total_curr = 0.0;
            node* b_compute = new node[total_vars]();
            matrix_mul_vec(total_vars, A, x, b_compute);
            for(int i = 0; i < n_nodes; ++i) {
                max_err = max(max_err, fabs(b[i].volt_re - b_compute[i].volt_re));
                max_total_curr = max(max_total_curr, max(fabs(b[i].volt_re), fabs(b_compute[i].volt_re)));
            }
            for (int i = 0; i < n_nodes; ++i) {
                max_volt_change = max(max_volt_change, fabs(x[i].volt_re - x_last[i].volt_re));
                max_volt = max(max_volt, max(fabs(x[i].volt_re), fabs(x_last[i].volt_re)));
            }
            for (int i = n_nodes; i < total_vars; ++i) {
                max_curr_change = max(max_curr_change, fabs(x[i].volt_re - x_last[i].volt_re));
                max_curr = max(max_volt, max(fabs(x[i].volt_re), fabs(x_last[i].volt_re)));
            }
            delete[] b_compute;

            if (max_err < RELTOL * max_total_curr + ABSTOL &&
                max_volt_change < RELTOL * max_volt + VNTOL &&
                max_curr_change < RELTOL * max_curr + ABSTOL) {
                converged = true;
                break;
            }
            for (int i = 0; i < total_vars; ++i) x_last[i] = x[i];
            iter++;
        }
        if (!converged) {
            cerr << "Warning: DC nonlinear iteration did not converge after " << MAX_ITER << " steps!" << endl;
        }
        delete[] x_last;
    }

    dc_voltages.clear();
    dc_voltages.resize(n_nodes);
    for (int i = 0; i < n_nodes; ++i) dc_voltages[i] = x[i].volt_re;

    if (print_output) {
        out << "=== DC Operating Point ===" << endl;
        out << "Node  Voltage (V)" << endl;
        out << "----  -----------" << endl;
        for (int i = 1; i <= n_nodes; ++i) {
            out << "V" << i << "    " << fixed << setprecision(6) << x[i-1].volt_re << " V" << endl;
        }
        out << "GND    0.000000 V" << endl;
    }

    for (int i = 0; i < total_vars; ++i) delete[] A[i];
    delete[] A; delete[] b; delete[] x;
}

// ==================== AC Single Frequency ====================
void solve_ac_single_frequency(double freq, ostream& out, bool with_header) {
    solve_dc_operating_point(cout, false);

    int max_node = 0;
    for (auto& v : voltage_sources) max_node = max(max_node, max(v.node1, v.node2));
    for (auto& i : current_sources) max_node = max(max_node, max(i.node1, i.node2));
    for (auto& r : resistors)       max_node = max(max_node, max(r.node1, r.node2));
    for (auto& c : capacitors)      max_node = max(max_node, max(c.node1, c.node2));
    for (auto& l : inductors)       max_node = max(max_node, max(l.node1, l.node2));
    for (auto& e : vcvs)            max_node = max(max_node, max(e.node1, e.node2));
    for (auto& g : vccs)            max_node = max(max_node, max(g.node1, g.node2));
    for (auto& f : cccs)            max_node = max(max_node, max(f.node1, f.node2));
    for (auto& h : ccvs)            max_node = max(max_node, max(h.node1, h.node2));

    int n_nodes = max_node;
    int n_vs = voltage_sources.size() + inductors.size() + vcvs.size() + ccvs.size();
    int total_vars = n_nodes + n_vs;
    if (total_vars == 0) { out << "No nodes." << endl; return; }

    node** A = new node*[total_vars];
    for (int i = 0; i < total_vars; ++i) A[i] = new node[total_vars]();
    node* b = new node[total_vars]();
    node* x = new node[total_vars]();

    build_ac_mna(n_nodes, n_vs, freq, "", 0.0, A, b);
    for (int i = 0; i < n_nodes; ++i) A[i][i] = A[i][i] + node(GMIN,0);
    gauss_elimination(total_vars, A, b, x);

    if (with_header) {
        out << "AC Single Frequency Analysis at f = " << freq << " Hz" << endl;
        out << "Node\tReal(V)\tImag(V)" << endl;
    }
    for (int i = 1; i <= n_nodes; ++i) {
        out << "V" << i << "\t" 
            << format_double(x[i-1].volt_re) << "\t" 
            << format_double(x[i-1].volt_im) << endl;
    }

    for (int i = 0; i < total_vars; ++i) delete[] A[i];
    delete[] A; delete[] b; delete[] x;
}

// ==================== AC Sweep ====================
void solve_ac_sweep(double start, double end, int points, int sweep_type,
                    const string& keep_name, double amplitude, ostream& out) {
    solve_dc_operating_point(cout, false);

    vector<double> freqs = generate_freq_points(start, end, points, sweep_type);
    if (freqs.empty()) { out << "Error generating frequency points." << endl; return; }

    int max_node = 0;
    for (auto& v : voltage_sources) max_node = max(max_node, max(v.node1, v.node2));
    for (auto& i : current_sources) max_node = max(max_node, max(i.node1, i.node2));
    for (auto& r : resistors)       max_node = max(max_node, max(r.node1, r.node2));
    for (auto& c : capacitors)      max_node = max(max_node, max(c.node1, c.node2));
    for (auto& l : inductors)       max_node = max(max_node, max(l.node1, l.node2));
    for (auto& e : vcvs)            max_node = max(max_node, max(e.node1, e.node2));
    for (auto& g : vccs)            max_node = max(max_node, max(g.node1, g.node2));
    for (auto& f : cccs)            max_node = max(max_node, max(f.node1, f.node2));
    for (auto& h : ccvs)            max_node = max(max_node, max(h.node1, h.node2));

    int n_nodes = max_node;
    int n_vs = voltage_sources.size() + inductors.size() + vcvs.size() + ccvs.size();
    int total_vars = n_nodes + n_vs;
    if (total_vars == 0) { out << "No nodes." << endl; return; }

    node** A = new node*[total_vars];
    for (int i = 0; i < total_vars; ++i) A[i] = new node[total_vars]();
    node* b = new node[total_vars]();
    node* x = new node[total_vars]();

    out << "Freq(Hz)";
    for (int i = 1; i <= n_nodes; ++i) out << "\tV" << i << "_real\tV" << i << "_imag";
    out << endl;

    for (double f : freqs) {
        for (int i = 0; i < total_vars; ++i) {
            fill(A[i], A[i] + total_vars, node(0,0));
            b[i] = node(0,0);
            x[i] = node(0,0);
        }
        build_ac_mna(n_nodes, n_vs, f, keep_name, amplitude, A, b);
        for (int i = 0; i < n_nodes; ++i) A[i][i] = A[i][i] + node(GMIN,0);
        gauss_elimination(total_vars, A, b, x);

        out << fixed << setprecision(3) << f;
        for (int i = 0; i < n_nodes; ++i) {
            out << "\t" << format_double(x[i].volt_re) 
                << "\t" << format_double(x[i].volt_im);
        }
        out << endl;
    }

    for (int i = 0; i < total_vars; ++i) delete[] A[i];
    delete[] A; delete[] b; delete[] x;
}

// ==================== Main ====================
int main() {
    while (true) {
        cout << "\n=== Main Menu ===" << endl;
        cout << "1. Read netlist from file" << endl;
        cout << "2. Exit" << endl;
        cout << "Choose: ";
        int choice = read_int("");

        if (choice == 2) break;
        if (choice != 1) continue;

        string filename;
        cout << "Enter filename: ";
        cin >> filename;
        ifstream fin(filename);
        if (!fin) {
            cerr << "Cannot open file: " << filename << endl;
            continue;
        }
        current_filename = filename;
        voltage_sources.clear(); current_sources.clear();
        resistors.clear(); capacitors.clear(); inductors.clear();
        diodes.clear(); vcvs.clear(); vccs.clear(); cccs.clear(); ccvs.clear();
        read_netlist(fin);
        fin.close();

        string base = get_basename(filename);

        while (true) {
            cout << "\n--- Circuit Menu ---" << endl;
            cout << "1. DC analysis" << endl;
            cout << "2. AC single frequency analysis" << endl;
            cout << "3. AC sweep analysis" << endl;
            cout << "4. Return to main menu" << endl;
            cout << "Choose: ";
            int sub = read_int("");

            if (sub == 4) break;

            switch (sub) {
                case 1: {
                    string outname = base + "DC.simout";
                    ofstream fout(outname);
                    if (fout) {
                        solve_dc_operating_point(fout);
                        fout.close();
                        cout << "DC results written to " << outname << endl;
                    } else {
                        cerr << "Cannot write to " << outname << endl;
                    }
                    solve_dc_operating_point(cout);
                    break;
                }
                case 2: {
                    cout << "Enter frequency (Hz): ";
                    double freq = read_double_with_units(cin);
                    string outname = base + "AC.simout";
                    ofstream fout(outname);
                    if (fout) {
                        fout << freq << endl;
                        solve_ac_single_frequency(freq, fout, false);
                        fout.close();
                        cout << "AC single frequency results written to " << outname << endl;
                    } else {
                        cerr << "Cannot write to " << outname << endl;
                    }
                    solve_ac_single_frequency(freq, cout, true);
                    break;
                }
                case 3: {
                    string keep_name;
                    double amp;
                    double start, end;
                    int sweep_type, points;

                    bool valid_source = false;
                    while (!valid_source) {
                        cout << "Enter active source name (e.g., V1 or I1): ";
                        cin >> keep_name;
                        for (auto& v : voltage_sources) {
                            if (v.name == keep_name && v.type == AC_SOURCE) {
                                valid_source = true;
                                break;
                            }
                        }
                        if (!valid_source) {
                            for (auto& i : current_sources) {
                                if (i.name == keep_name && i.type == AC_SOURCE) {
                                    valid_source = true;
                                    break;
                                }
                            }
                        }
                        if (!valid_source) {
                            cout << "Error: Source '" << keep_name << "' not found or not AC type. Please re-enter." << endl;
                        }
                    }

                    cout << "Enter amplitude for this source: ";
                    amp = read_double_with_units(cin);
                    cout << "Enter start and end frequencies (Hz): ";
                    start = read_double_with_units(cin);
                    end = read_double_with_units(cin);
                    sweep_type = read_int("Sweep type: 1-Linear, 2-Decade (points per decade): ");
                    points = read_int("Enter number of points (linear: total points, decade: points per decade): ");

                    string outname = base + "ACsweep.simout";
                    ofstream fout(outname);
                    if (fout) {
                        solve_ac_sweep(start, end, points, sweep_type, keep_name, amp, fout);
                        fout.close();
                        cout << "AC sweep results written to " << outname << endl;
                    } else {
                        cerr << "Cannot write to " << outname << endl;
                    }
                    break;
                }
                default:
                    cout << "Invalid choice." << endl;
            }
        }
    }
    return 0;
}