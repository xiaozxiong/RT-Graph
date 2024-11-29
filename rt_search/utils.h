#pragma once

#include <random>

std::default_random_engine generator;

void UniformDistribution(float *data,int n,float minn=0.0f,float maxn=100000.0f){
    std::uniform_real_distribution<float> distribution(minn,maxn);
    for(int i=0;i<n;i++){
        float val=distribution(generator);
        data[i]=val;
    }
}

void NormalDistribution(float *data,int n,float minn=0.0f,float maxn=100000.0f){
    float mean=(minn+maxn)/2; //*
    float std=(maxn-minn)/50; //*
    std::normal_distribution<float> distribution(mean,std);
    for(int i=0;i<n;i++){
        float x=distribution(generator);
        while(x<minn||x>maxn) x=distribution(generator);
        data[i]=x;
    }
}

void ExponentialDistribution(float *data,int n,float minn=0.0f,float maxn=10000.0f){
    std::exponential_distribution<float> distribution(3.5); //* lambda
    float interval=maxn-minn;
    for(int i=0;i<n;i++){
        float x=distribution(generator);
        while(x>1.0) x=distribution(generator);
        data[i]=minn+interval*x;
    }
}

// void Check(int *rt_results,int *bs_results,int result_size,float *query,float *data){
//     bool flag=true;
//     for(int i=0;i<result_size;i++){
//         if(rt_results[i]!=bs_results[i]&&(int)abs(rt_results[i]-bs_results[i])>5){
//             flag=false;
//             printf("Query %d = %f: rt = %d (%f), bs = %d (%f)\n",i,query[i],rt_results[i],data[rt_results[i]],bs_results[i],data[bs_results[i]]);
//         }
//     }
//     if(flag) printf("Accepted\n");
// }

void Check(int *ans,int *res,int n){
    int cnt=0;
    for(int i=0;i<n;i++){
        if(ans[i]!=res[i]) cnt+=1;
    }
    if(cnt==0) printf("Accepted\n");
    else printf("Wrong = %d\n",cnt);
}

