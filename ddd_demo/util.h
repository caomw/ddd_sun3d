#include <ctime>


///////////////////////////////////////////////////////////////////////
// Timer
std::clock_t tic_toc_timer;
void tic() {
  tic_toc_timer = clock();
}
void toc() {
  std::clock_t toc_timer = clock();
  printf("Elapsed time is %f seconds.\n", double(toc_timer - tic_toc_timer) / CLOCKS_PER_SEC);
}