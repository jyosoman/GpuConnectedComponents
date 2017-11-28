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
 *       Please report any issues to Jyothish Soman (jyothish@students.iiit.ac.in)
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


/*
 *
 *    Function to speedup the selection process in the first iteration
 *    The ancestor tree is initialized to the add the edge from larger edge to its smaller neighbour in this method.
 *    The process is random and each edge performs this task independently.
 *    select_winner_init
 *
 */
struct ed{
    long long int x;
};

typedef struct ed edge;

struct grp{
    int num_e,num_n;
    int**neigh,*deg;
};

typedef struct grp my_graph;


__global__ 
void select_winner_init(int* an,edge *ed_list,int num_e,int num_n,int*flag,char*mark){
    int a,b,x,y,mn,mx;
    long long int t;
    a=blockIdx.y*gridDim.x+blockIdx.x;
    b=threadIdx.x;
    a=a*512+b;
    if(a<num_e){
        t=ed_list[a].x;
        x=(int)t & 0xFFFFFFFF;
        y=(int)(t>>32);

        mx=x>y?x:y;
        mn=x+y-mx;
        an[mx]=mn;
    }
    return;
}

/*
   Function to hook from higher valued tree to lower valued tree. For details, read the PPL Paper or LSPP paper or my master's thesis.
   Following greener's algorithm, there are two iterations, one from lower valued edges to higher values edges
   and the second iteration goes vice versa. The performance of this is largely related to the input.

 */   



__global__ 
void select_winner2(int* an,edge *ed_list,int num_e,int num_n,int*flag,char*mark){
    int a,b,x,y,a_x,a_y,mn,mx;
    long long int t;
    a=blockIdx.y*gridDim.x+blockIdx.x;
    b=threadIdx.x;
    __shared__ int s_flag;
    a=a*512+b;
    if(b==1)
        s_flag=0;
    __syncthreads();
    if(a<num_e){
        if(mark[a]==0){
            t=ed_list[a].x;
            x=(int)t & 0xFFFFFFFF;
            y=(int)(t>>32);

            a_x=an[x];
            a_y=an[y];
            mx=a_x>a_y?a_x:a_y;
            mn=a_x+a_y-mx;
            if(mn==mx){
                mark[a]=-1;
            }
            else{
                an[mn]=mx;
                s_flag=1;
            }
        }
    }
    __syncthreads();
    if(b==1){
        if(s_flag==1){
            *flag=1;
        }
    }
    return;
}


/*
   Function to hook from lower valued to higher valued trees. 



 */   
__global__ 
void select_winner(int* an,edge *ed_list,int num_e,int num_n,int*flag,char*mark){
    int a,b,x,y,a_x,a_y,mn,mx;
    long long int t;
    a=blockIdx.y*gridDim.x+blockIdx.x;
    b=threadIdx.x;
    __shared__ int s_flag;
    a=a*512+b;
    if(b==1)
        s_flag=0;
    __syncthreads();
    if(a<num_e){
        if(mark[a]==0){
            t=ed_list[a].x;
            x=(int)t & 0xFFFFFFFF;
            y=(int)(t>>32);

            a_x=an[x];
            a_y=an[y];
            mx=a_x>a_y?a_x:a_y;
            mn=a_x+a_y-mx;
            if(mn==mx){
                mark[a]=-1;
            }
            else{
                an[mx]=mn;
                s_flag=1;
            }
        }
    }
    __syncthreads();
    if(b==1){
        if(s_flag==1){
            *flag=1;
        }
    }
    return;
}




__global__ 
void p_jump(int num_n,int* an,int *flag){
    int a,b,x,y;
    a=blockIdx.y*gridDim.x+blockIdx.x;		
    b=threadIdx.x;
    a=a*512+b;
    __shared__ int s_f;
    if(a>=num_n)
        return;
    if(b==1){
        s_f=0;
    }
    __syncthreads();
    if(a<num_n){
        y=an[a];
        x=an[y];
        if(x!=y){
            s_f=1;
            an[a]=x;
        }
    }
    if(b==1){
        if(s_f==1){
            *flag=1;
        }
    }
}


/*
   Function to do a masked jump
   Nodes are either root nodes or leaf nodes. Leaf nodes are directly connected to the root nodes, hence do not
   need to jump itertively. Once root nodes have reascertained the new root nodes, the leaf nodes can just jump once


 */
__global__ 
void p_jump_masked(int num_n,int* an,int *flag,char*mask){
    int a,b,x,y;
    a=blockIdx.y*gridDim.x+blockIdx.x;
    b=threadIdx.x;
    a=a*512+b;
    __shared__ int s_f;
    if(a>=num_n)
        return;
    if(b==1){
        s_f=0;
    }

    __syncthreads();
    if(mask[a]==0){
        y=an[a];
        x=an[y];
        if(x!=y){
            s_f=1;
            an[a]=x;
        }
        else{
            mask[a]=-1;
        }
    }
    if(b==1){
        if(s_f==1){
            *flag=1;
        }
    }
}

/*

   Function for pointer jumping in the tree, the tree height is shortened by this method.
   Here the assumption is that all the nodes are root nodes, or not known whether they are leaf nodes.
   Works well in the early iterations

 */

__global__ 
void p_jump_unmasked(int num_n,int* an,char *mask){
    int a,b,x,y;
    a=blockIdx.y*gridDim.x+blockIdx.x;
    b=threadIdx.x;
    a=a*512+b;
    if(a>=num_n)
        return;
    __syncthreads();
    if(mask[a]==1){
        y=an[a];
        x=an[y];
        an[a]=x;
    }
}

/*
   Function to create self pointing tree.
 */
__global__ 
void update_an(int*an,int num_n){
    int a,b;
    a=blockIdx.y*gridDim.x+blockIdx.x;		
    b=threadIdx.x;
    a=a*512+b;
    if(a>=num_n)
        return;
    an[a]=a;

    return;
}

/*
   Function to initialize each edge as a clean copy. 
 */
__global__ 
void	update_mark(char *mark,int num_e){
    int j;
    j=blockIdx.y*gridDim.x+blockIdx.x;
    j=j*512+threadIdx.x;
    if(j>=num_e)
        return;
    mark[j]=0;
}

/*
   Function to check if each node is the parent of itself or not and to update it as a leaf or root node

 */

__global__ 
void update_mask(char *mask,int n,int *an){
    int j;
    j=blockIdx.y*gridDim.x+blockIdx.x;
    j=j*512+threadIdx.x;
    if(j>=n)
        return;
    mask[j]=an[j]==j?0:1;
    return;
}


