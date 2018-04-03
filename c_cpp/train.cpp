#include "hmm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG

void Forward(HMM *hmm, int oSeqLength,int** oSeqAddr, double*** localProbabilityAlphaAddr, double* finalProbabilityAddr){
    int* oSeq = *oSeqAddr;
    double finalProbability = *finalProbabilityAddr;
    double** localProbabilityAlpha = *localProbabilityAlphaAddr;

    int stateIndexI;//for pre
    int stateIndexJ;//for new
    int timeIndexT;
    double middleSum;

    for(stateIndexI=0; stateIndexI < hmm->state_num; stateIndexI++){
        localProbabilityAlpha[0][stateIndexI]=hmm->initial[stateIndexI] * hmm->observation[oSeq[0]][stateIndexI];
    }

    for(timeIndexT=0; timeIndexT < oSeqLength-1; timeIndexT++){
        for(stateIndexJ=0; stateIndexJ < hmm->state_num; stateIndexJ++){
            middleSum=0.0;
            for(stateIndexI=0; stateIndexI < hmm->state_num; stateIndexI++){
                middleSum+=localProbabilityAlpha[timeIndexT][stateIndexI] * hmm->transition[stateIndexI][stateIndexJ];//[I][J] or [J][I] depend on the read-file and hmm.h
            }
            localProbabilityAlpha[timeIndexT+1][stateIndexJ] = middleSum * (hmm->observation[oSeq[timeIndexT+1]][stateIndexJ]);
        }
    }

    finalProbability = 0.0;
    for(stateIndexI = 0; stateIndexI < hmm->state_num; stateIndexI++){
        finalProbability += localProbabilityAlpha[timeIndexT][stateIndexI];
    }

    return;
}

void Backward(HMM *hmm, int oSeqLength, int** oSeqAddr, double*** localProbabilityBetaAddr, double* finalProbabilityAddr){
    int* oSeq = *oSeqAddr;
    double** localProbabilityBeta = *localProbabilityBetaAddr;
    double finalProbability = *finalProbabilityAddr;
    int stateIndexI;//for pre
    int stateIndexJ;//for new
    int timeIndexT;
    double middleSum;

    for(stateIndexI = 0; stateIndexI < hmm->state_num; stateIndexI++){
        localProbabilityBeta[oSeqLength][stateIndexI] = 1.0;
    }

    for(timeIndexT=oSeqLength-1; timeIndexT>=0; timeIndexT--){
        for(stateIndexI=0; stateIndexI < hmm->state_num; stateIndexI++){
            middleSum = 0.0;
            for(stateIndexJ=0; stateIndexJ < hmm->state_num; stateIndexJ++){
                middleSum += hmm->transition[stateIndexI][stateIndexJ] * (hmm->observation[oSeq[timeIndexT+1]][stateIndexJ]) * localProbabilityBeta[timeIndexT+1][stateIndexJ];
            }
            localProbabilityBeta[timeIndexT][stateIndexI] = middleSum;
        }
    }

    finalProbability = 0.0;
    for(stateIndexI=0; stateIndexI < hmm->state_num; stateIndexI++){
        finalProbability += localProbabilityBeta[0][stateIndexI];
    }

    return;
}

void ComputeGamma(HMM* hmm, int oSeqLength, double*** alphaAddr, double*** betaAddr, double*** gammaAddr){
    double** alpha = *alphaAddr;
    double** beta = *betaAddr;
    double** gamma = *gammaAddr;

    int stateIndexI;
    int timeIndexT;
    double sum;


    for(timeIndexT=0; timeIndexT < oSeqLength; timeIndexT++){
        sum = 0.0;
        for(stateIndexI=0; stateIndexI < hmm->state_num; stateIndexI++){
            sum+=alpha[timeIndexT][stateIndexI] * beta[timeIndexT][stateIndexI];
        }

        for(stateIndexI=0; stateIndexI < oSeqLength; stateIndexI++){
            gamma[timeIndexT][stateIndexI] = alpha[timeIndexT][stateIndexI] * beta[timeIndexT][stateIndexI] / sum;
        }
    }

    return;
}


void ComputeXi(HMM* hmm, int oSeqLength, int** oSeqAddr, double*** alphaAddr, double*** betaAddr, double**** xiAddr){
    int* oSeq = *oSeqAddr;
    double** alpha = *alphaAddr;
    double** beta = *betaAddr;
    double*** xi = *xiAddr;

    int stateIndexI;
    int stateIndexJ;
    int timeIndexT;
    double sum;

    for(timeIndexT=0; timeIndexT < oSeqLength; timeIndexT++){
        sum = 0.0;
        for(stateIndexI=0; stateIndexI < hmm->state_num; stateIndexI++){
            for(stateIndexJ=0; stateIndexJ < hmm->state_num; stateIndexJ++){
                sum+=alpha[timeIndexT][stateIndexI] * hmm->transition[stateIndexI][stateIndexJ] * (hmm->observation[oSeq[timeIndexT+1]][stateIndexJ]) * beta[timeIndexT+1][stateIndexJ];
            }
        }

        for(stateIndexI=0; stateIndexI < hmm->state_num; stateIndexI++){
            for(stateIndexJ=0; stateIndexJ < hmm->state_num; stateIndexJ++){
                xi[timeIndexT][stateIndexI][stateIndexJ] = alpha[timeIndexT][stateIndexI] * hmm->transition[stateIndexI][stateIndexJ] * (hmm->observation[oSeq[timeIndexT+1]][stateIndexJ]) * beta[timeIndexT+1][stateIndexJ] / sum;
            }
        }
    }

    return;
}


void NewInitial(HMM* hmm, double** newInitialAddr, double*** gammaAddr){
    double* newInitial = *newInitialAddr;
    double** gamma = *gammaAddr;
    int timeIndexT = 0;
    int stateIndexI;
    for(stateIndexI = 0; stateIndexI < hmm->state_num; stateIndexI++){
        newInitial[stateIndexI] = gamma[timeIndexT][stateIndexI];
    }

    return;
}

void NewTransition(HMM* hmm, int oSeqLength, double*** newTransitionAddr, double*** gammaAddr, double**** xiAddr){
    double** newTransition = *newTransitionAddr;
    double** gamma = *gammaAddr;
    double*** xi = *xiAddr;
    
    int stateIndexI;
    int stateIndexJ;
    int timeIndexT;
    double xiSum;
    double gammaSum;

    for(stateIndexI=0; stateIndexI < hmm->state_num; stateIndexI++){
        for(stateIndexJ=0; stateIndexJ < hmm->state_num; stateIndexJ++){
            xiSum = 0.0;
            for(timeIndexT=0; timeIndexT < oSeqLength-1; timeIndexT++){
                xiSum+=xi[timeIndexT][stateIndexI][stateIndexJ];
            }

            gammaSum = 0.0;
            for(timeIndexT=0; timeIndexT < oSeqLength-1; timeIndexT++){
                gammaSum+=gamma[timeIndexT][stateIndexI];
            }

            newTransition[stateIndexI][stateIndexJ]=xiSum/gammaSum;
        }
    }

    return;
}

void NewObservation(HMM* hmm, int oSeqLength, int** oSeqAddr, double*** newObservationAddr, double*** gammaAddr){
    int* oSeq = *oSeqAddr;
    double** newObservation = *newObservationAddr;
    double** gamma = * gammaAddr;
    
    int stateIndexJ;
    int obStateIndexK;
    int timeIndexT;
    double gammaSum;
    double oEqKGammaSum;


    for(obStateIndexK=0; obStateIndexK < oSeqLength; obStateIndexK++){
        for(stateIndexJ=0; stateIndexJ < hmm->state_num; stateIndexJ++){
            gammaSum = 0.0;
            for(timeIndexT=0; timeIndexT < oSeqLength; timeIndexT++){
                gammaSum+=gamma[timeIndexT][stateIndexJ];
            }

            oEqKGammaSum = 0.0;
            for(timeIndexT=0; (timeIndexT < oSeqLength) && (oSeq[timeIndexT]==obStateIndexK); timeIndexT++){
                oEqKGammaSum+=gamma[timeIndexT][stateIndexJ];
            }

            newObservation[obStateIndexK][stateIndexJ]=oEqKGammaSum/gammaSum;
        }
    }


    return;
}

void New1DimArray(double ** new1DimArray, int A){
    *new1DimArray = (double *)malloc(A * sizeof(double));
    return;
}

void New2DimArray(double*** new2DimArray, int A, int B){

    fprintf(stderr, "info: %d: ", __LINE__);
    *new2DimArray = (double **)malloc(A * sizeof(double *));
    fprintf(stderr, "info: %d: ", __LINE__);
    for(int a=0; a < A; a++){
        fprintf(stderr, "info: %d: ", __LINE__);
        (*new2DimArray)[a] = (double *)malloc(B * sizeof(double));
    }
    return;
}
 
void New3DimArray(double**** new3DimArray, int A, int B, int C){
    *new3DimArray = (double***)malloc(A * sizeof(double**));
    for(int a=0; a < A; a++){
        (*new3DimArray)[a] = (double**)malloc(B * sizeof(double*));
        for(int b=0; b < B; b++){
            (*new3DimArray)[a][b] = (double*)malloc(C * sizeof(double));
        }
    }
    return;
}


void Test2DimArray(double*** new2DimArray, int A, int B){
    double testValue=0.0;
    fprintf(stderr, "info: %d: ", __LINE__);

    for(int a=0; a < A; a++){
        for(int b=0; b < B; b++){
        fprintf(stderr, "info: %d: ", __LINE__);

            (*new2DimArray)[a][b] = testValue;
            testValue+=1;
        }
    }
    fprintf(stderr, "info: %d: ", __LINE__);

    for(int a=0; a < A; a++){
        for(int b=0; b < B; b++){
            printf("%lf ", (*new2DimArray)[a][b]);
        }
    }
    printf("\n");
    return;
}


void WriteHmm(HMM* hmm, HMM* newHmm, double** newInitialAddr, double*** newTransitionAddr, double*** newObservationAddr){
    strcpy( newHmm->model_name, hmm->model_name );
    newHmm->observ_num = hmm->observ_num;
    newHmm->state_num = hmm->state_num;
    for( int i = 0 ; i < hmm->state_num ; i++ ){
        newHmm->initial[i] = (*newInitialAddr)[i];
    }
    for( int i = 0 ; i < hmm->state_num ; i++ ){
        for( int j = 0 ; j < hmm->state_num ; j++ ){
            newHmm->transition[i][j] = (*newTransitionAddr)[i][j];
        }
    }
    for( int i = 0 ; i < hmm->observ_num ; i++ ){
        for( int j = 0 ; j < hmm->state_num ; j++ ){
            newHmm->observation[i][j] = (*newObservationAddr)[i][j];
        }
    }      

    return;
}

void NewHmm(HMM* hmm, int oSeqLength, int** oSeqAddr, HMM* newHmm, 
    double*** localProbabilityAlphaAddr,
    double* finalProbabilityAlphaAddr,
    double*** localProbabilityBetaAddr,
    double* finalProbabilityBetaAddr,
    double*** gammaAddr,
    double**** xiAddr,
    double** newInitialAddr,
    double*** newTransitionAddr,
    double*** newObservationAddr
    ){
    // int* oSeq = *oSeqAddr;

    // double** localProbabilityAlpha;
    // New2DimArray(&localProbabilityAlpha,oSeqLength,hmm->state_num);
    // double finalProbabilityAlpha;
    Forward(hmm, oSeqLength, oSeqAddr, localProbabilityAlphaAddr, finalProbabilityAlphaAddr);

    // double** localProbabilityBeta;
    // New2DimArray(&localProbabilityBeta,oSeqLength,hmm->state_num);
    // double finalProbabilityBeta;
    Backward(hmm, oSeqLength, oSeqAddr, localProbabilityBetaAddr, finalProbabilityBetaAddr);

    // double** gamma;
    // New2DimArray(&gamma, oSeqLength, hmm->state_num);
    ComputeGamma(hmm, oSeqLength, localProbabilityAlphaAddr, localProbabilityBetaAddr, gammaAddr);

    // double*** xi;
    // New3DimArray(&xi,oSeqLength,hmm->state_num,hmm->state_num);
    ComputeXi(hmm, oSeqLength, oSeqAddr, localProbabilityAlphaAddr, localProbabilityBetaAddr, xiAddr);

    // double* newInitial;
    // New1DimArray(&newInitial, hmm->state_num);
    NewInitial(hmm, newInitialAddr, gammaAddr);

    // double** newTransition;
    // New2DimArray(&newTransition, hmm->state_num, hmm->state_num);
    NewTransition(hmm, oSeqLength, newTransitionAddr, gammaAddr, xiAddr);

    // double** newObservation;
    // New2DimArray(&newObservation, oSeqLength, hmm->state_num);
    NewObservation(hmm, oSeqLength, oSeqAddr, newObservationAddr, gammaAddr);

    WriteHmm(hmm,newHmm, newInitialAddr, newTransitionAddr, newObservationAddr);

    return;
}


void BaulmWelch(HMM* hmm, int oSeqLength, int** oSeqAddr, int iteration){
    
    double** localProbabilityAlpha;
    New2DimArray(&localProbabilityAlpha,oSeqLength,hmm->state_num);
    double finalProbabilityAlpha;

    double** localProbabilityBeta;
    New2DimArray(&localProbabilityBeta,oSeqLength,hmm->state_num);
    double finalProbabilityBeta;

    double** gamma;
    New2DimArray(&gamma, oSeqLength, hmm->state_num);

    double*** xi;
    New3DimArray(&xi,oSeqLength,hmm->state_num,hmm->state_num);

    double* newInitial;
    New1DimArray(&newInitial, hmm->state_num);

    double** newTransition;
    New2DimArray(&newTransition, hmm->state_num, hmm->state_num);

    double** newObservation;
    New2DimArray(&newObservation, oSeqLength, hmm->state_num);

    HMM nextHMM;
    
    HMM* oldHmm;
    HMM* newHMM;
    
    oldHmm = hmm;
    newHMM = &nextHMM;

    for(int iterationIndex=0; iterationIndex < iteration; iterationIndex++){
        NewHmm(oldHmm,oSeqLength,oSeqAddr, newHMM,
        &localProbabilityAlpha,
        &finalProbabilityAlpha,
        &localProbabilityBeta, 
        &finalProbabilityBeta,
        &gamma,
        &xi,
        &newInitial,
        &newTransition,
        &newObservation
        );

        oldHmm = newHMM;
        newHMM = hmm;
    }

    return;
}





int main(int argc, char* argv[]){
    if(argv[1]!=NULL){
        int iteration = atoi(argv[1]);
    }
    // char* model_init = argv[2];
    // char* seq_model = argv[3];
    // char* model = argv[4];




    return 0;
}