/****************************************************************************************
 *       CONNECTED COMPONENTS ON THE GPU                                                        
 *       ==============================
 *
 *
 *
 *       Copyright (c) 2010 International Institute of Information Technology,
 *       Hyderabad.
 *       All rights reserved.
 *
 *       Permission to use, copy, modify and distribute this software and its
 *       documentation for research purpose is hereby granted without fee,
 *       provided that the above copyright notice and this permission notice appear
 *       in all copies of this software and that you do not sell the software.
 *
 *       THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,
 *       EXPRESS, IMPLIED OR OTHERWISE.
 *
 *       Please report any issues to Jyothish Soman (first.last at gmail)
 *
 *       Please cite following paper, if you use this software for research purpose
 *
 *       "Fast GPU Algorithms for Graph Connectivity, Jyothish Soman, K. Kothapalli, 
 *       and P. J. Narayanan, in Proc. of Large Scale Parallel Processing, 
 *       IPDPS Workshops, 2010.
 *
 *
 *
 *
 *       Created by Jyothish Soman
 *											
 ****************************************************************************************/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include<time.h>


#include<helper_cuda.h>
#include<helper_functions.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include"conn.cu"

/*
 *
 *   Function to load graph to the CPU memory.
 *   load_graph
 *   input is currently hard coded as edge.txt and parameters.txt
 *   edge.txt contains the edges, in the format n1 n2
 *   parameters.txt contains the number of edges and number of nodes respectively
 *   Do note that this is a connected components for undirected graphs, each edge is undirected. 
 *   The input here is assumed to be undirected edges, no duplicate edges assumed, but not necessary.   
 *
 *   #TODO make this code more cleaner and natural, add a config file or a shell wrapper to make this more user friendly
 *
 *
 *
 */

void load_graph(edge** ed_list,int*num_n,int*num_e){
    FILE*fp,*fp2;
    edge*edl;
    int i,j,x,y,a;
    long long int v;
    fp=fopen("edge.txt","r");
    fp2=fopen("parameters.txt","r");
    fscanf(fp2,"%d%d",&i,&j);
    *ed_list=(edge*)calloc(i,sizeof(edge));
    edl=*ed_list;
    if(edl==NULL){
        printf("Insufficient memory, data lost");
        exit(0);
    }
    for(a=0;a<i;a++){
        fscanf(fp,"%d%d",&x,&y);
        x=x-1;
        y=y-1;
        v=0;
        v=(long long int)x;
        v=v<<32;
        v+=(long long int) y;
        edl[a].x=v;
    }
    *num_n=j;
    *num_e=i;
    fclose(fp);
    fclose(fp2);
    return;
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{

    findCudaDevice(argc,(const char**) argv);

    edge* ed_list,*d_ed_list;
    int num_n,num_e,nnx,nny,nex,ney;	
//    unsigned int timer1 = 0;
//    checkCudaErrors( cutCreateTimer( &timer1));
//    checkCudaErrors( cutStartTimer( timer1));


    load_graph(&ed_list,&num_n,&num_e);

    int flag,*d_winner,*d_an;
    int *d_flag,*an;
    char*d_mark,*mark;
    char*mask;


    int num_threads,num_blocks_n,num_blocks_e;
    num_threads=512;
    num_blocks_n=(num_n/512)+1;
    num_blocks_e=(num_e/512)+1;
    nny=(num_blocks_n/1000)+1;
    nnx=1000;
    nex=(num_blocks_e/1000)+1;
    ney=1000;
    dim3  grid_n( nnx, nny);
    dim3  grid_e( nex, ney);
    dim3  threads( num_threads, 1);

    an=(int*)calloc(num_n,sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&d_mark,num_e*sizeof(char)));
    checkCudaErrors(cudaMalloc((void**)&mask,num_e*sizeof(char)));
    checkCudaErrors(cudaMalloc((void**)&d_winner,num_n*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_an,num_n*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_ed_list,num_e*sizeof(edge)));
    checkCudaErrors(cudaMalloc((void**)&d_flag,sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_ed_list,ed_list,num_e*sizeof(edge),cudaMemcpyHostToDevice));

    

    //   Finished intializing space for the program, ideally timing should be from here.

    clock_t t = clock();
    


    update_mark<<< grid_e,threads>>>(d_mark,num_e);
    update_an<<< grid_n,threads>>>(d_an,num_n);
    cudaThreadSynchronize();

    cudaThreadSynchronize();


    //First round of select winner


    select_winner_init<<< grid_e,threads>>>(d_an,d_ed_list,num_e,num_n,d_flag,d_mark);
    cudaThreadSynchronize();


//    CUT_CHECK_ERROR("Kernel execution failed");


    do{
        flag=0;
        checkCudaErrors(cudaMemcpy(d_flag,&flag,sizeof(int),cudaMemcpyHostToDevice));
        p_jump<<< grid_n,threads>>>(num_n,d_an,d_flag);
        cudaThreadSynchronize();

//        CUT_CHECK_ERROR("Kernel execution failed");
        checkCudaErrors(cudaMemcpy(&flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost));
    }while(flag);

    //main code starts
    //
    update_mask<<< grid_n,threads>>>(mask,num_n,d_an);
    int lpc=1;
    do{
        flag=0;				
        checkCudaErrors(cudaMemcpy(d_flag,&flag,sizeof(int),cudaMemcpyHostToDevice));
        if(lpc!=0){
            select_winner<<< grid_e,threads>>>(d_an,d_ed_list,num_e,num_n,d_flag,d_mark);
            lpc++;
            lpc=lpc%4;
        }
        else{

            select_winner2<<< grid_e,threads>>>(d_an,d_ed_list,num_e,num_n,d_flag,d_mark);
            lpc=0;
        }
        cudaThreadSynchronize();

 //       CUT_CHECK_ERROR("Kernel execution failed");
        checkCudaErrors(cudaMemcpy(&flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost));
        if(flag==0){
            break;
        }

//        CUT_CHECK_ERROR("Kernel execution failed");

        int flg;
        do{
            flg=0;
            checkCudaErrors(cudaMemcpy(d_flag,&flg,sizeof(int),cudaMemcpyHostToDevice));
            p_jump_masked<<< grid_n,threads>>>(num_n,d_an,d_flag,mask);
            cudaThreadSynchronize();

//            CUT_CHECK_ERROR("Kernel execution failed");
            checkCudaErrors(cudaMemcpy(&flg,d_flag,sizeof(int),cudaMemcpyDeviceToHost));
        }while(flg);

        p_jump_unmasked<<< grid_n,threads>>>(num_n,d_an,mask);
        cudaThreadSynchronize();
//        CUT_CHECK_ERROR("Kernel execution failed");

        update_mask<<< grid_n,threads>>>(mask,num_n,d_an);
//        CUT_CHECK_ERROR("Kernel execution failed");
        cudaThreadSynchronize();
    }while(flag);
    t = clock() - t;
    /* checkCudaErrors( cutStopTimer( timer)); */
    /* printf( "%f\n", cutGetTimerValue( timer)); */
    /* checkCudaErrors( cutDeleteTimer( timer)); */
    printf ("Time required for computing connected components on the graph is: %f seconds.\n",((float)t)/CLOCKS_PER_SEC);
    
    
    mark=(char*)calloc(num_e,sizeof(char));
    //end of main loop
    checkCudaErrors(cudaMemcpy(an,d_an,num_n*sizeof(int),cudaMemcpyDeviceToHost));
    int j,cnt=0;
    for(j=0;j<num_n;j++){
        if(an[j]==j){
            cnt++;
        }
    }

    printf("The number of components=%d\n",cnt);
    free(an);
    free(mark);
    checkCudaErrors(cudaFree(d_an));
    checkCudaErrors(cudaFree(d_ed_list));
    checkCudaErrors(cudaFree(d_flag));
    checkCudaErrors(cudaFree(d_mark));
}
