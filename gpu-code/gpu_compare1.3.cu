#include <iostream>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <sys/time.h>

using namespace std;

#ifndef LeafN
#define LeafN 99322
#endif

#ifndef OrderN
#define OrderN 99321
#endif

struct Abd
{
    //char name[20];
    string name;
    float data[LeafN];
};
struct HashNode
{
    string name;
    unsigned int time;
};
struct CompData
{
    float Dist_1[OrderN];
    float Dist_2[OrderN];
    int Order_1[OrderN];
    int Order_2[OrderN];
    int Order_d[OrderN];
    int Id[LeafN];
};
unsigned int BKDRHash(string str0)
{
    char * str;
    int len=str0.length();
    str=new char[len];
    strcpy(str,str0.c_str());
    unsigned int seed=131 ;// 31 131 1313 13131 131313 etc..
    unsigned int hash=0 ;
    while(*str)
    {
        hash=hash*seed+(*str++);
    }
    return(hash);
}
int get_compData(CompData *cd)
{
    cout<<"begin to copy"<<endl;
    memcpy(cd->Dist_1, Dist_1, sizeof(float) * OrderN);
    memcpy(cd->Dist_2, Dist_2, sizeof(float) * OrderN);
    memcpy(cd->Order_1, Order_1, sizeof(int) * OrderN);
    memcpy(cd->Order_2, Order_2, sizeof(int) * OrderN);
    memcpy(cd->Order_d, Order_d, sizeof(int) * OrderN);
    memcpy(cd->Id, Id, sizeof(int) * OrderN);
    return 1;
}


#ifndef GPUC
#define GPUC

__global__ void gpu_Calc_sim(CompData *cd, Abd *v_Abd_1, float *g_hashmap, unsigned int * g_abdIndex, float *results);
class gpu_compare
{
  public:
    gpu_compare();
    int sendData(CompData * cd,Abd * abd1, Abd * abd2,int count, int version = 2);
        //please malloc or new memory space for these data, and don't free these memory untill you retrive the results
        //count means the number of Abd2
        //default to use second generation compare algorithm
    int act();
    //int sendCompData(CompData * cd);
    int getResult(float * putResult); 
    /*
    error table  0 for all right, 1 for no enough memory
    2 for error state for example, you ask the gpu_compare to act before send data to it
    */

  private:
    int getInfo();
    int malloc(int count);
    int maxMem, deviceID, countCore,state,count;
    unsigned int *abdIndex,* g_abdIndex;
    cudaDeviceProp deviceProp;
    size_t freeMem, totalMem;
    CompData *compData, *g_compData;
    Abd *abd1, *abd2, *g_abd1, *g_abd2;
    float *results, *g_results;
    int version;
    unsigned int queryTime;
    int hashSize,usedHash;
    HashNode * hashmap;
    float * g_hashmap;
};

/*int gpu_compare::sendCompData(CompData *cd)
{
    this->compData = cd;
    cudaMalloc((void **)&this->g_compData,sizeof(CompData));
    cudaMemcpy(this->g_compData,cd,sizeof(CompData),cudaMemcpyHostToDevice);
    this->state+=0.5;
    return 0;
}*/
int gpu_compare::getInfo() {
    cudaGetDeviceProperties(&this->deviceProp, this->deviceID);
    this->maxMem = this->deviceProp.totalGlobalMem;
    this->countCore = this->deviceProp.multiProcessorCount;
    cudaMemGetInfo(&this->freeMem, &this->totalMem);
    return 0;
}
gpu_compare::gpu_compare()
{
    size_t a;
    int hashSize;
    HashNode * hashmap;
    this->queryTime=0;
    cudaChooseDevice(&this->deviceID, &this->deviceProp);
    this->getInfo();
    a=this->freeMem- sizeof(CompData)- sizeof(Abd)*2000;
    this->hashSize=a/ ((sizeof(Abd)+ sizeof(float))*1.1);
    hashSize=this->hashSize;
    this->usedHash=0;
    this->hashmap=new HashNode[this->hashSize];
    hashmap=this->hashmap;
    for(int t=0;t<hashSize;t++)
    {
        hashmap[t].time=0;
    }
    cudaMalloc((void **)&this->g_hashmap, sizeof(float)*LeafN*this->hashSize);
    this->state=1;
}
int gpu_compare::sendData(CompData * cd,Abd *abd1, Abd *abd2,int count,int version)
{
    this->queryTime++;
        /*
        1 for no enough memory
        2 for state error
        */
    if (!(this->state==1 || this->state==1.5 ||this->state==4))
    {
        cout<<"gpu_compare is not in right state!\n";
        return 2;
    }
    //cout<<"begin to copy data"<<endl;

    this->count=count;
    this->abd1 = abd1;
    this->abd2 = abd2;
    this->version = version;
    this->compData = cd;
    
    this->getInfo();
    cout<<"free memory"<<this->freeMem/1024/1024/1024<<endl;
    //time
    if (totalMem<sizeof(float)*(LeafN*(count+1)+count)*1.1)
    {
        cout<<"not enough gpu memory, will update later to support it\n";
        return 1;
    }
    else
    {
        // malloc the memory
        cudaMalloc((void **)&this->g_compData,sizeof(CompData));
        cudaMalloc((void **)&this->g_abd1,sizeof(Abd));
        cudaMalloc((void **)&this->g_abdIndex,sizeof(unsigned int)*count);
        cudaMalloc((void **)&this->g_results,sizeof(float)*count);
    }
    this->abdIndex=new unsigned int[count];
    this->malloc(count);
    //send memory from main mamory to GPU memory
    //cout<<"finish copy tree to gpu"<<endl;
    cudaMemcpy(this->g_compData,cd,sizeof(CompData),cudaMemcpyHostToDevice);
    cudaMemcpy(this->g_abd1,abd1,sizeof(Abd),cudaMemcpyHostToDevice);
    cudaMemcpy(this->g_abdIndex,abdIndex,sizeof(unsigned int)*count,cudaMemcpyHostToDevice);
    this->state=2;
    return 0;
}
int gpu_compare::malloc(int count)
{
    HashNode *hashmap;
    unsigned int offset,offset2,*abdIndex;
    int crashTime,flag,timer;
    hashmap=this->hashmap;
    unsigned int queryTime=this->queryTime;
    int hashSize=this->hashSize;
    Abd aBuffer,*abd2;
    abd2=this->abd2;
    HashNode hBuffer;
    abdIndex=this->abdIndex;
    timer=0;
    for(int i=0;i<count;i++)
    {
        aBuffer=abd2[i];
        offset=BKDRHash(aBuffer.name)%hashSize;
        hBuffer=hashmap[offset];
        if(hBuffer.name==aBuffer.name)
        {
            hashmap[offset].time=queryTime;
            abdIndex[i]=offset;
            timer++;
        } else
        {
            offset2=offset;
            flag=1;
            for (int t = 0 ;t < 5||flag ==1; t++)
            {
                offset2+=t*t;
                if(hashmap[offset2].time==0)
                {
                    offset=offset2;
                    abdIndex[i]=offset;
                    hashmap[offset].time=queryTime;
                    cudaMemcpy(this->g_hashmap+offset,abd2[i].data, sizeof(float) * LeafN,cudaMemcpyHostToDevice);
                    this->usedHash++;
                    flag=2;
                    break;
                }
                if(hashmap[offset2].name==aBuffer.name)
                {
                    hashmap[offset].time=queryTime;
                    abdIndex[i]=offset2;
                    timer++;
                    flag=2;
                    break;
                }
                if(hashmap[offset2].time<queryTime&&flag)
                {    
                    offset=offset2;
                    flag=0;
                }
            }
            if(flag!=2)
            {
                abdIndex[i]=offset;
                hashmap[offset].time=queryTime;
                cudaMemcpy(this->g_hashmap+offset,abd2[i].data, sizeof(float) * LeafN,cudaMemcpyHostToDevice);
            }
        }
    }
    cout<<"cache miss rate:"<<1-timer/count<<endl;
}
int gpu_compare::act()
{
    if(this->state!=2)
    {
        cout<<"gpu_compare state error\n";
        cout<<"current state:"<<this->state<<endl;
        return 2;
    }

    gpu_Calc_sim<<<this->count, 1>>>(this->g_compData, this->g_abd1, this->g_hashmap,this->g_abdIndex, this->g_results);
    
    this->state=3;
    return 0;
}

int gpu_compare::getResult(float *putResult)
{
    if(this->state!=3)
    {
        cout<<"gpu_compare state error\n";
        return 2;
    }
    //this->results=new float[count];
    //cout<<"finish calculating"<<endl;
    struct timeval tv_begin,tv_end;
    gettimeofday(&tv_begin,NULL);
    cudaMemcpy(putResult, this->g_results, sizeof(float) * count,cudaMemcpyDeviceToHost);
    gettimeofday(&tv_end,NULL);

    double copyBackTime = double(tv_end.tv_sec-tv_begin.tv_sec)*1000000+double(tv_end.tv_usec-tv_begin.tv_usec);
    cout<<"copyBackTime:"<<copyBackTime<<endl;
    /*for(int i=0;i<count;i++)
    {
        printf("the similarity is: %f",putResult[i]);
    }*/
    this-> state=4;
    //putResult=this->results;

    //cout<<"finish copy results to CPU"<<endl;
    cudaFree(this->g_abd1);
    cudaFree(this->g_results);
    return 0;
}
__global__ void gpu_Calc_sim(CompData *cd, Abd *v_Abd_1, float *g_hashmap, unsigned int * g_abdIndex, float *results)
{
    //process memory data
    float *Dist_1;
    float *Dist_2;
    int *Order_1;
    int *Order_2;
    int *Order_d;
    int *Id;
    Dist_1 = cd->Dist_1;
    Dist_2 = cd->Dist_2;
    Order_1 = cd->Order_1;
    Order_2 = cd->Order_2;
    Order_d = cd->Order_d;
    Id = cd->Id;
    //change offset of each parameters

    //const Meta_Result * buffer=(Meta_Result * )v_buffer+blockIdx.x;// don't know wether it's used
    unsigned int offset;
    offset=g_abdIndex[blockIdx.x];
    const float *Abd_2 = (float *) g_hashmap+offset;
    const float *Abd_1 = (float *) &(v_Abd_1->data);

    //start origin data
    float Reg_1[70];
    float Reg_2[70];
    float Reg_abs[70];
    
    float total = 0;
    float total2=0;
    float total3=0;
    int root;
    
    for(int i = 0; i < OrderN; i++){
        //cout<<"i:"<<endl;
        
        int order_1 = Order_1[i];
        int order_2 = Order_2[i];
        int order_d = Order_d[i] + 70;
        
        float dist_1 = 1- Dist_1[i];
        float dist_2 = 1- Dist_2[i];
        
        float c1_1;
        float c1_2;
        
        float c2_1;
        float c2_2;

        float abs_1;
        float abs_2;
                        
        if (order_1 >= 0){
                    
                c1_1 = Abd_1[order_1];
                c1_2 = Abd_2[order_1];
                abs_1=abs(Abd_1[order_1]- Abd_2[order_1]) * 0.5;
                }
        else {
            c1_1 = Reg_1[order_1 + 70];
            c1_2 = Reg_2[order_1 + 70];
            abs_1=Reg_abs[order_1 + 70];
            }
        
        if (order_2 >= 0){
                    
                    c2_1 = Abd_1[order_2];
                    c2_2 = Abd_2[order_2];
                    abs_2=abs(Abd_1[order_2]-Abd_2[order_2]) * 0.5;
                    
                    }
        else {
            c2_1 = Reg_1[order_2 + 70];
            c2_2 = Reg_2[order_2 + 70];
            abs_2 = Reg_abs[order_2 + 70];
            }
        //min
        float min_1 = (c1_1 < c1_2)?c1_1:c1_2;
        float min_2 = (c2_1 < c2_2)?c2_1:c2_2;
        
        total += min_1;
        total2 += abs(c1_1-c1_2);

        
        total += min_2;
        total2 += abs(c2_1-c2_2);

        
        /*if(abs(c2_1-c2_2) !=0  || abs(c1_1-c1_2) !=0)
        {
        cout<<c1_1<<"-"<<c1_2<<"="<<c1_1-c1_2<<" "<<abs(c1_1-c1_2)<<endl;
        cout<<c2_1<<"-"<<c2_2<<"="<<c2_1-c2_2<<" "<<abs(c2_1-c2_2)<<endl;
        cout<<"total2:"<<total2<<endl;
        }
        */

        
        
        //reduce
        Reg_1[order_d] = (c1_1 - min_1) * dist_1 + (c2_1 - min_2) * dist_2;
        Reg_2[order_d] = (c1_2 - min_1) * dist_1 + (c2_2 - min_2) * dist_2;
        Reg_abs[order_d]= abs_1*dist_1 + abs_2*dist_2;
        
        root = order_d;
        }
    
      total += (Reg_1[root] < Reg_2[root])?Reg_1[root]:Reg_2[root];
      //cout<<"total:"<<total<<endl;
      //total2 += abs(Reg_1[root]-Reg_2[root]);
      //cout<<"second score:"<<total2<<endl;
      //cout<<"third score:"<<100-Reg_abs[root]<<endl;
      //cout<<total<<"\t"<<100-Reg_abs[root]<<endl;
      
      //return total;
      
      results[blockIdx.x]=total;
      //100-Reg_abs[root];

      return;
      
      }


#endif
