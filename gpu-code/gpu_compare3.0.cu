#include <iostream>

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
struct CompData
{
    float Dist_1[OrderN];
    float Dist_2[OrderN];
    int Order_1[OrderN];
    int Order_2[OrderN];
    int Order_d[OrderN];
    int Id[LeafN];
};
struct GPUInfo
{
    int computSum, deviceID;
    cudaDeviceProp deviceProp;
};


#ifndef GPUC
#define GPUC

__global__ void gpu_Calc_sim(CompData *cd, Abd *v_Abd_1, Abd *v_Abd_2, float *results, int count);
class gpu_compare
{
public:
    gpu_compare::gpu_compare(){};
    gpu_compare(int* gpuSelect,int count);
    int initial_gpu(int* gpuSelect,int count);
    int sendData(CompData * cd, Abd * abd1, Abd * abd2,int count, int version = 2);
    //please malloc or new memory space for these data, and don't free these memory untill you retrive the results
    //count means the number of Abd2
    //default to use second generation compare algorithm
    int act();
    int getResult(float * putResult);
    int action(CompData *cd, Abd *abd1, Abd *abd2,int count, int version, float * getResult );
    /*
    error table  0 for all right, 1 for no enough memory
    2 for error state for example, you ask the gpu_compare to act befor send data to it
    */
private:
    int state,count,*gpuSelect;
    CompData *compData, **g_compData;
    Abd *abd1, *abd2, **g_abd1, **g_abd2;
    float *results, **g_results,*calBility,*calTime;
    int bestID,gpuCount,version,compFlag,*alloc,*offset;
    cudaDeviceProp bestProp;
    GPUInfo *gpuInfo;
};



gpu_compare::gpu_compare(int* gpuSelect,int count)
{
    this->gpuCount=count;
    this->gpuSelect=new int[count];
    memcpy(this->gpuSelect,gpuSelect, sizeof(int)*count);
    this->offset=new int[count];
    int *gS=this->gpuSelect;
    if(count==1) {
        int deviceID;
        this->gpuInfo=new GPUInfo[1];
        cudaChooseDevice(&this->gpuInfo->deviceID,&this->gpuInfo->deviceProp);
        if(*gS!=this->gpuInfo->deviceID)
            cout<<"\nThe GPU you selected is ID "<<*gS<<", but CUDA thinks the ID:"<<this->gpuInfo->deviceID<<" is best"
                <<"maybe you can change it to get best experience\n";
        this->bestID=this->gpuInfo->deviceID;
        this->bestProp=this->gpuInfo->deviceProp;
    }
    else {
        this->gpuInfo = new GPUInfo[gS[this->gpuCount]];
        this->alloc=new int[this->gpuCount];
        this->calBility=new float[this->gpuCount];
        this->calTime=new float[this->gpuCount];
        for (int i = this->gpuCount-1; i >=0 ; i--) {
            cudaGetDeviceProperties(&this->gpuInfo[i].deviceProp, gS[i]);
            this->gpuInfo[i].computSum=this->gpuInfo[i].deviceProp.warpSize*this->gpuInfo[i].deviceProp.multiProcessorCount;
            cudaChooseDevice(&this->bestID, &this->bestProp);
            this->alloc[i]=1;
            this->calBility[i]=1;
            this->calTime[i]=1;
        }
    }
    this->compFlag=1;
    this->state=1;
}

int gpu_compare::initial_gpu(int* gpuSelect,int count){
    this->gpuCount=count;
    this->gpuSelect=new int[count];
    memcpy(this->gpuSelect,gpuSelect, sizeof(int)*count);
    this->offset=new int[count];
    int *gS=this->gpuSelect;
    if(count==1) {
        int deviceID;
        this->gpuInfo=new GPUInfo[1];
        cudaChooseDevice(&this->gpuInfo->deviceID,&this->gpuInfo->deviceProp);
        if(*gS!=this->gpuInfo->deviceID)
            cout<<"\nThe GPU you selected is ID "<<*gS<<", but CUDA thinks the ID:"<<this->gpuInfo->deviceID<<" is best"
                <<"maybe you can change it to get best experience\n";
        this->bestID=this->gpuInfo->deviceID;
        this->bestProp=this->gpuInfo->deviceProp;
    }
    else {
        this->gpuInfo = new GPUInfo[this->gpuCount];
        this->alloc=new int[this->gpuCount];
        this->calBility=new float[this->gpuCount];
        this->calTime=new float[this->gpuCount];
        for (int i = this->gpuCount-1; i >=0 ; i--) {
            cudaGetDeviceProperties(&this->gpuInfo[i].deviceProp, gS[i]);
            this->gpuInfo[i].computSum=this->gpuInfo[i].deviceProp.warpSize*this->gpuInfo[i].deviceProp.multiProcessorCount;
            cudaChooseDevice(&this->bestID, &this->bestProp);
            this->alloc[i]=1;
            this->calBility[i]=1;
            this->calTime[i]=1;
        }
    }
    this->compFlag=1;
    this->state=1;
    return 1;

}


int gpu_compare::sendData(CompData *cd, Abd *abd1, Abd *abd2,int count, int version)
{
    int Sum=0;
    /*
    1 for no enough memory
    2 for state error
    */
    int *gS=this->gpuSelect,gC=this->gpuCount,*alc=this->alloc;
    if (this->state!=1)
    {
        cout<<"gpu_compare is not in right state!\n";
        return 2;
    }

    this->count=count;
    this->compData = cd;
    this->abd1 = abd1;
    this->abd2 = abd2;
    this->version = version;
    for (int i = 0; i <this->gpuCount ; i++) {
        if (this->gpuInfo[i].deviceProp.totalGlobalMem<sizeof(float)*(LeafN*(count+1)+count)*1.1) {
            cout << "no enough gpu memory in " << this->gpuInfo[i].deviceProp.name << endl;
            if(this->gpuCount>1)
                cout << "you are using multi-GPU model, which require each GPU have enough memory to contain all data."
                    <<" We may update in the future, and now, you can block this GPU or switch to single-GPU model";
        }
    }
        this->g_abd1 = new Abd *[gC];
        this->g_abd2 = new Abd *[gC];
        this->g_results = new float *[gC];
        this->g_compData = new CompData *[gC];
    for (int i = this->gpuCount-1; i >=0 ; i--) {
        Sum+=this->gpuInfo[i].computSum;
    }
    int offset=0;
    for (int i = 0; i <gC ; ++i) {
        alc[i]=this->gpuInfo[i].computSum*(count/Sum+1);
        if(alc[i]+offset>=this->count)
            alc[i]=this->count-offset;
        cudaSetDevice(gS[i]);
        // malloc the memory
        cudaMalloc((void **) &this->g_abd1[i], sizeof(Abd));
        cudaMalloc((void **) &this->g_abd2[i], sizeof(Abd) * alc[i]);
        cudaMalloc((void **) &this->g_results[i], sizeof(float) * alc[i]);
        if (this->compFlag) {
            cudaMalloc((void **) &this->g_compData[i], sizeof(CompData));
            compFlag = 0;
        }//send memory from main mamory to GPU memory
        cudaMemcpy(this->g_compData[i],cd,sizeof(CompData),cudaMemcpyHostToDevice);
        cudaMemcpy(this->g_abd1[i],abd1,sizeof(Abd),cudaMemcpyHostToDevice);
        cudaMemcpy(this->g_abd2[i],abd2+offset,sizeof(Abd)*alc[i],cudaMemcpyHostToDevice);
        this->offset[i]=offset;
        offset+=alc[i];
    }
    this->state=2;
    return 0;
}

int gpu_compare::act()
{
    if(this->state!=2){
        cout<<"gpu_compare state error\n";
        return 2;
    }
    int *gS=this->gpuSelect,gC=this->gpuCount,*alc=this->alloc;
    for (int i = 0; i <gC ; ++i) {
        cudaSetDevice(gS[i]);
        gpu_Calc_sim<<<alc[i]/this->gpuInfo[i].deviceProp.warpSize+1,this->gpuInfo[i].deviceProp.warpSize>>>
            (this->g_compData[i], this->g_abd1[i], this->g_abd2[i], this->g_results[i],alc[i]);
    }
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
    int *gS=this->gpuSelect,gC=this->gpuCount,*alc=this->alloc;
    this->results=putResult;
    //this->results=new float[count];
    for (int i = 0; i <gC ; ++i) {
        cudaSetDevice(gS[i]);
        cudaMemcpy(this->results+this->offset[i],
            this->g_results[i],
            sizeof(float) * alc[i],
            cudaMemcpyDeviceToHost);
        cudaFree(this->g_abd1[i]);
        cudaFree(this->g_abd2[i]);
        cudaFree(this->g_results[i]);
    }
    for(int i=0;i<count;i++)
    {
        cout<<i<<endl;
        printf("the similarity is: %f",putResult[i]);
    }
    this->state=1;
    return 0;
}


int gpu_compare::action(CompData *cd, Abd *abd1, Abd *abd2,int count, int version, float * getResult )
{
    this->sendData(cd,abd1,abd2,count,version);
    this->act();
    this->getResult(getResult);
    return 1;
}



__global__ void gpu_Calc_sim(CompData *cd, Abd *v_Abd_1, Abd *v_Abd_2, float *results,int count)
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

    int offset;
    offset=blockIdx.x*blockDim.x+threadIdx.x;
    if(offset>=count)
        return ;
    //const Meta_Result * buffer=(Meta_Result * )v_buffer+blockIdx.x;// don't know wether it's used
    v_Abd_2=(Abd *)v_Abd_2 +offset;
    const float *Abd_2 = v_Abd_2->data;
    const float *Abd_1 = v_Abd_1->data;

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

    results[offset]=total;
    //100-Reg_abs[root];
    return;
}
#endif

