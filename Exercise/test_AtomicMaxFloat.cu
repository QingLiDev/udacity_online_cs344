#include <iostream>
#include <cstdlib>
#include <algorithm>

using namespace std;

int main(int argc, char **argv) {
  srand(time(NULL));

  const int ARRAY_SIZE = 10;
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++) h_in[i] = float(i);
  random_shuffle(&h_in[0], &h_in[ARRAY_SIZE]);
  //http://stackoverflow.com/questions/14720134/is-it-possible-to-random-shuffle-an-array-of-int-elements
  for (int i = 0; i < ARRAY_SIZE; i++) cout << h_in[i] << endl;
}
