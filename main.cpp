#include <bits/stdc++.h>
#include <cmath>
using namespace std;

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

// ==================== Function Declarations ====================
double norm2(const node& z);
void gauss_elimination(int n, node** A, node* b, node* x);
void read_netlist(istream& in);
void build_dc_mna(int n_nodes, int n_vs, node** A, node* b);
void build_ac_mna(int n_nodes, int n_vs, double freq, const string& keep_name,
                  double amp, node** A, node* b);
string get_basename(const string& filename);
vector<double> generate_freq_points(double start, double end, int points, int type);
void solve_dc_operating_point(ostream& out = cout);
void solve_ac_single_frequency(double freq, ostream& out = cout, bool with_header = true);
void solve_ac_sweep(double start, double end, int points, int sweep_type,
                    const string& keep_name, double amplitude, ostream& out = cout);

// Helper for formatted output
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

vector<VoltageSource> voltage_sources;
vector<CurrentSource> current_sources;
vector<Resistor> resistors;
vector<Capacitor> capacitors;
vector<Inductor> inductors;

// ==================== Utility Functions ====================
double norm2(const node& z) {
    return z.volt_re * z.volt_re + z.volt_im * z.volt_im;
}

void gauss_elimination(int n, node** A, node* b, node* x) {
    using namespace std::chrono;
    auto start = steady_clock::now();
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
    auto end = steady_clock::now();
    duration<double> elapsed = end - start;
    double sec = elapsed.count();
    if (sec < 1e-9) cout << "Time: " << sec * 1e12 << " ps" << endl;
    else if (sec < 1e-6) cout << "Time: " << sec * 1e9 << " ns" << endl;
    else if (sec < 1e-3) cout << "Time: " << sec * 1e6 << " us" << endl;
    else if (sec < 1.0) cout << "Time: " << sec * 1e3 << " ms" << endl;
    else cout << "Time: " << sec << " s" << endl;
}

// Format a double: use scientific notation for values with magnitude < 1e-3 (but not zero),
// otherwise use fixed-point with specified precision (default 8).
string format_double(double val, int precision) {
    ostringstream oss;
    if (fabs(val) < 1e-3 && val != 0.0) {
        oss << scientific << setprecision(precision) << val;
    } else {
        oss << fixed << setprecision(precision) << val;
    }
    return oss.str();
}

// Parse a string with engineering suffix (e.g., 1k, 2.5M, 100u)
double parse_value(const string& s) {
    if (s.empty()) return 0.0;
    // Supports both upper and lower case for convenience (U/u = micro, K/k = kilo)
    // Added n, p, f, a for nano, pico, femto, atto
    static const unordered_map<char, double> prefix = {
        {'u', 1e-6}, {'U', 1e-6},
        {'m', 1e-3},
        {'k', 1e3}, {'K', 1e3},
        {'M', 1e6},
        {'G', 1e9},
        {'T', 1e12},
        {'n', 1e-9}, {'N', 1e-9},   // nano
        {'p', 1e-12}, {'P', 1e-12}, // pico
        {'f', 1e-15}, {'F', 1e-15}, // femto
        {'a', 1e-18}, {'A', 1e-18}  // atto
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

// Read a double from standard input with full engineering unit support
double read_double_with_units(istream& in) {
    string s;
    in >> s;
    return parse_value(s);
}

// ==================== Safe integer input ====================
// Reads an integer from stdin with error recovery
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

// ==================== Netlist Reader (from any stream) ====================
void read_netlist(istream& in) {
    in >> c_size;
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
                    vs.ac_phase = parse_value(ph); // degrees
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
            default:
                cerr << "Error: unknown component type '" << type << "'" << endl;
        }
    }
    // Check for ground
    bool has_gnd = false;
    for (auto& v : voltage_sources) if (v.node1==0 || v.node2==0) has_gnd = true;
    for (auto& i : current_sources) if (i.node1==0 || i.node2==0) has_gnd = true;
    for (auto& r : resistors)       if (r.node1==0 || r.node2==0) has_gnd = true;
    for (auto& c : capacitors)      if (c.node1==0 || c.node2==0) has_gnd = true;
    for (auto& l : inductors)       if (l.node1==0 || l.node2==0) has_gnd = true;
    if (!has_gnd) cerr << "Warning: No component connected to GND (node 0). Circuit may be floating!" << endl;
}

// ==================== Get Base Filename ====================
string get_basename(const string& filename) {
    size_t dot = filename.find_last_of('.');
    if (dot != string::npos) return filename.substr(0, dot);
    return filename;
}

// ==================== Generate Frequency Points ====================
vector<double> generate_freq_points(double start, double end, int points, int type) {
    vector<double> freqs;
    if (type == 1) { // Linear
        if (points <= 1) { freqs.push_back(start); return freqs; }
        double step = (end - start) / (points - 1);
        for (int i = 0; i < points; ++i)
            freqs.push_back(start + i * step);
    } else { // Decade (type=2) – points = number of intervals per decade
        if (start <= 0 || end <= 0 || points <= 0) {
            cerr << "Error: start, end must be positive and points > 0 for log sweep." << endl;
            return freqs;
        }
        double ratio = pow(10.0, 1.0 / points); // frequency multiplier per step
        double f = start;
        const double eps = 1e-12;
        while (f <= end * (1.0 + eps)) {
            freqs.push_back(f);
            f *= ratio;
        }
        // Ensure the end frequency is included
        if (freqs.back() < end - eps) {
            freqs.push_back(end);
        } else if (fabs(freqs.back() - end) > eps) {
            // Replace the last point with end if it overshoots slightly
            freqs.back() = end;
        }
    }
    return freqs;
}

// ==================== DC MNA Construction ====================
void build_dc_mna(int n_nodes, int n_vs, node** A, node* b) {
    auto idx = [&](int node) { return (node == 0) ? -1 : node - 1; };
    // Resistors
    for (auto& r : resistors) {
        double g = 1.0 / r.value;
        int p = idx(r.node1), q = idx(r.node2);
        if (p != -1) { A[p][p] = A[p][p] + node(g,0); if(q!=-1) A[p][q] = A[p][q] - node(g,0); }
        if (q != -1) { A[q][q] = A[q][q] + node(g,0); if(p!=-1) A[q][p] = A[q][p] - node(g,0); }
    }
    // Current sources (DC only)
    for (auto& i : current_sources) {
        double I = i.dc_value;
        int p = idx(i.node1), q = idx(i.node2);
        if (p != -1) b[p] = b[p] + node(I,0);
        if (q != -1) b[q] = b[q] - node(I,0);
    }
    // Voltage sources (DC) + Inductors (short circuit)
    int vs_idx = n_nodes;
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
    for (auto& l : inductors) {
        int p = idx(l.node1), q = idx(l.node2);
        if (p != -1) A[p][vs_idx] = A[p][vs_idx] + node(1,0);
        if (q != -1) A[q][vs_idx] = A[q][vs_idx] - node(1,0);
        if (p != -1) A[vs_idx][p] = A[vs_idx][p] + node(1,0);
        if (q != -1) A[vs_idx][q] = A[vs_idx][q] - node(1,0);
        b[vs_idx] = node(0,0);
        vs_idx++;
    }
    // Capacitors ignored (open circuit)
}

// ==================== AC MNA Construction ====================
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
    // Inductors Y = -j/(ωL)
    for (auto& l : inductors) {
        double y_im = -1.0 / (omega * l.value);
        int p = idx(l.node1), q = idx(l.node2);
        if (p != -1) { A[p][p] = A[p][p] + node(0, y_im); if(q!=-1) A[p][q] = A[p][q] - node(0, y_im); }
        if (q != -1) { A[q][q] = A[q][q] + node(0, y_im); if(p!=-1) A[q][p] = A[q][p] - node(0, y_im); }
    }

    // Current sources: if keep_name is empty, all AC sources are active;
    // otherwise, only the specified source (phase 0) with amplitude 'amp' is active
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

    // Voltage sources: each introduces a current variable
    int vs_idx = n_nodes;
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
    // Inductors already handled in admittance
}

// ==================== DC Analysis ====================
void solve_dc_operating_point(ostream& out) {
    int max_node = 0;
    for (auto& v : voltage_sources) max_node = max(max_node, max(v.node1, v.node2));
    for (auto& i : current_sources) max_node = max(max_node, max(i.node1, i.node2));
    for (auto& r : resistors)       max_node = max(max_node, max(r.node1, r.node2));
    for (auto& c : capacitors)      max_node = max(max_node, max(c.node1, c.node2));
    for (auto& l : inductors)       max_node = max(max_node, max(l.node1, l.node2));

    int n_nodes = max_node;
    int n_vs = voltage_sources.size() + inductors.size();
    int total_vars = n_nodes + n_vs;
    if (total_vars == 0) { out << "No nodes." << endl; return; }

    node** A = new node*[total_vars];
    for (int i = 0; i < total_vars; ++i) A[i] = new node[total_vars]();
    node* b = new node[total_vars]();
    node* x = new node[total_vars]();

    build_dc_mna(n_nodes, n_vs, A, b);
    for (int i = 0; i < n_nodes; ++i) A[i][i] = A[i][i] + node(GMIN, 0);

    gauss_elimination(total_vars, A, b, x);

    out << "=== DC Operating Point ===" << endl;
    out << "Node  Voltage (V)" << endl;
    out << "----  -----------" << endl;
    for (int i = 1; i <= n_nodes; ++i) {
        double v = x[i-1].volt_re;
        out << "V" << i << "    " << fixed << setprecision(6) << v << " V" << endl;
    }
    out << "GND    0.000000 V" << endl;

    for (int i = 0; i < total_vars; ++i) delete[] A[i];
    delete[] A; delete[] b; delete[] x;
}

// ==================== AC Single-Frequency Analysis ====================
void solve_ac_single_frequency(double freq, ostream& out, bool with_header) {
    int max_node = 0;
    for (auto& v : voltage_sources) max_node = max(max_node, max(v.node1, v.node2));
    for (auto& i : current_sources) max_node = max(max_node, max(i.node1, i.node2));
    for (auto& r : resistors)       max_node = max(max_node, max(r.node1, r.node2));
    for (auto& c : capacitors)      max_node = max(max_node, max(c.node1, c.node2));
    for (auto& l : inductors)       max_node = max(max_node, max(l.node1, l.node2));

    int n_nodes = max_node;
    int n_vs = voltage_sources.size();
    int total_vars = n_nodes + n_vs;
    if (total_vars == 0) { out << "No nodes." << endl; return; }

    node** A = new node*[total_vars];
    for (int i = 0; i < total_vars; ++i) A[i] = new node[total_vars]();
    node* b = new node[total_vars]();
    node* x = new node[total_vars]();

    // keep_name empty means all AC sources are retained
    build_ac_mna(n_nodes, n_vs, freq, "", 0.0, A, b);
    for (int i = 0; i < n_nodes; ++i) A[i][i] = A[i][i] + node(GMIN, 0);

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

// ==================== AC Sweep Analysis ====================
void solve_ac_sweep(double start, double end, int points, int sweep_type,
                    const string& keep_name, double amplitude, ostream& out) {
    vector<double> freqs = generate_freq_points(start, end, points, sweep_type);
    if (freqs.empty()) { out << "Error generating frequency points." << endl; return; }

    int max_node = 0;
    for (auto& v : voltage_sources) max_node = max(max_node, max(v.node1, v.node2));
    for (auto& i : current_sources) max_node = max(max_node, max(i.node1, i.node2));
    for (auto& r : resistors)       max_node = max(max_node, max(r.node1, r.node2));
    for (auto& c : capacitors)      max_node = max(max_node, max(c.node1, c.node2));
    for (auto& l : inductors)       max_node = max(max_node, max(l.node1, l.node2));

    int n_nodes = max_node;
    int n_vs = voltage_sources.size();
    int total_vars = n_nodes + n_vs;
    if (total_vars == 0) { out << "No nodes." << endl; return; }

    node** A = new node*[total_vars];
    for (int i = 0; i < total_vars; ++i) A[i] = new node[total_vars]();
    node* b = new node[total_vars]();
    node* x = new node[total_vars]();

    // Output header: Frequency + Real and Imag parts for each node
    out << "Freq(Hz)";
    for (int i = 1; i <= n_nodes; ++i) out << "\tV" << i << "_real\tV" << i << "_imag";
    out << endl;

    for (double f : freqs) {
        // Reset matrix and vector
        for (int i = 0; i < total_vars; ++i) {
            fill(A[i], A[i] + total_vars, node(0,0));
            b[i] = node(0,0);
            x[i] = node(0,0);
        }
        build_ac_mna(n_nodes, n_vs, f, keep_name, amplitude, A, b);
        for (int i = 0; i < n_nodes; ++i) A[i][i] = A[i][i] + node(GMIN, 0);
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

// ==================== Main Program ====================
int main() {
    while (true) {
        cout << "\n=== Main Menu ===" << endl;
        cout << "1. Read netlist from file" << endl;
        cout << "2. Exit" << endl;
        cout << "Choose: ";
        int choice = read_int("");  // prompt already printed

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
        // Clear previous data
        voltage_sources.clear(); current_sources.clear();
        resistors.clear(); capacitors.clear(); inductors.clear();
        read_netlist(fin);
        fin.close();

        string base = get_basename(filename);

        // Inner menu loop
        while (true) {
            cout << "\n--- Circuit Menu ---" << endl;
            cout << "1. DC analysis" << endl;
            cout << "2. AC single frequency analysis" << endl;
            cout << "3. AC sweep analysis" << endl;
            cout << "4. Return to main menu" << endl;
            cout << "Choose: ";
            int sub = read_int("");  // prompt already printed

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
                    // Also display on screen
                    solve_dc_operating_point(cout);
                    break;
                }
                case 2: {
                    cout << "Enter frequency (Hz): ";
                    double freq = read_double_with_units(cin);
                    string outname = base + "AC.simout";
                    ofstream fout(outname);
                    if (fout) {
                        // Write frequency to first line
                        fout << freq << endl;
                        // Output data, no header
                        solve_ac_single_frequency(freq, fout, false);
                        fout.close();
                        cout << "AC single frequency results written to " << outname << endl;
                    } else {
                        cerr << "Cannot write to " << outname << endl;
                    }
                    // Also display on screen (with header)
                    solve_ac_single_frequency(freq, cout, true);
                    break;
                }
                case 3: {
                    string keep_name;
                    double amp;
                    double start, end;
                    int sweep_type, points;

                    // Validate active source name
                    bool valid_source = false;
                    while (!valid_source) {
                        cout << "Enter active source name (e.g., V1 or I1): ";
                        cin >> keep_name;
                        // Check if it exists and is an AC source
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
                    // Do not output to screen
                    break;
                }
                default:
                    cout << "Invalid choice." << endl;
            }
        }
    }
    return 0;
}