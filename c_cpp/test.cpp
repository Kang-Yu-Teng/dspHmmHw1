#include "hmm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv){
    HMM hmm_initial;
	loadHMM( &hmm_initial, argv[1] );
	dumpHMM( stderr, &hmm_initial );

	printf("%f\n", log(1.5) ); // make sure the math library is included
	return 0;
}