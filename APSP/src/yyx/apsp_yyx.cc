/****************************************************************
* 1.OpenMP
* 2.Compiler optimizations : -O3 -ffast-math -mavx512f -march=native
* 3.SIMD
* 4.Loop tiling : block width = 64
* N.Other trivial optimizations...

*****************************************************************/


#include "graph.hh"
#include<immintrin.h>
#include<omp.h>

#include<algorithm>


#define BLOCK_WIDTH 64
#define likely(x) __builtin_expect(!!(x),1)


#define SIMD


Graph Graph::apsp() {
    Graph result(*this);
    int width=result.vertex_num();


  
    //decide the number of threads
    if(width==128){
        omp_set_num_threads(4); 
    }
    else if(width==512){
        omp_set_num_threads(16);
    }
    else if(width==1024){
        omp_set_num_threads(32);
    }
    else{
        omp_set_num_threads(64);
    } 


    int * tmp=(int *)aligned_alloc(64,width*width*sizeof(int));
    auto tmp_result = (int (*) [width])tmp;
  
    #pragma omp parallel 
    {   

        //copy the result(x,y) to tmp_result[x][y]
        #pragma omp for collapse(2)
        for(int i=0;i<width;++i){
            for(int j=0;j<width;++j){
                tmp_result[i][j]=result(i,j);
            }
        }


        //Partition the task into diffierent phases
        for(int p=0;p<width/BLOCK_WIDTH;++p){
            int start=p*BLOCK_WIDTH;
            int end=(p+1)*BLOCK_WIDTH;


            
            //stage 1: Process the independent block first.          
            for(int k=start;k<end;++k){
                #pragma omp for schedule(static,1) 
                for(int i=start;i<end;++i){
                    #ifndef SIMD
                    int tmp=tmp_result[i][k];
                    for(int j=start;j<end;++j)    tmp_result[i][j]=std::min(tmp_result[i][j],tmp+tmp_result[k][j]);      
                    #else
                    __m512i _tmp = _mm512_set1_epi32(tmp_result[i][k]);
                    #ifndef UNROLL
                    for(int j=start;j<end;j+=16){
                        __m512i _res_kj = _mm512_load_epi32(&tmp_result[k][j]);
                        __m512i _res_ij = _mm512_load_epi32(&tmp_result[i][j]);
                        __m512i _tmp_res_kj = _mm512_add_epi32(_tmp,_res_kj);
                        _res_ij = _mm512_min_epi32(_res_ij,_tmp_res_kj);
                        _mm512_store_epi32(&tmp_result[i][j],_res_ij);
                    } 
                    #else
                    __m512i _res_kj0=_mm512_load_epi32(&tmp_result[k][start]);
                    __m512i _res_kj1=_mm512_load_epi32(&tmp_result[k][start+16]);
                    __m512i _res_kj2=_mm512_load_epi32(&tmp_result[k][start+32]);
                    __m512i _res_kj3=_mm512_load_epi32(&tmp_result[k][start+48]);
                    
                    __m512i _tmp_res_kj0 = _mm512_add_epi32(_tmp,_res_kj0);  
                    __m512i _tmp_res_kj1 = _mm512_add_epi32(_tmp,_res_kj1);
                    __m512i _tmp_res_kj2 = _mm512_add_epi32(_tmp,_res_kj2);
                    __m512i _tmp_res_kj3 = _mm512_add_epi32(_tmp,_res_kj3);

                    __m512i _res_ij0 = _mm512_load_epi32(&tmp_result[i][start]);
                    __m512i _res_ij1 = _mm512_load_epi32(&tmp_result[i][start+16]);
                    __m512i _res_ij2 = _mm512_load_epi32(&tmp_result[i][start+32]);
                    __m512i _res_ij3 = _mm512_load_epi32(&tmp_result[i][start+48]);

                    _res_ij0 = _mm512_min_epi32(_res_ij0,_tmp_res_kj0);
                    _res_ij1 = _mm512_min_epi32(_res_ij1,_tmp_res_kj1);
                    _res_ij2 = _mm512_min_epi32(_res_ij2,_tmp_res_kj2);
                    _res_ij3 = _mm512_min_epi32(_res_ij3,_tmp_res_kj3);
                    
                    _mm512_store_epi32(&tmp_result[i][start],_res_ij0);
                    _mm512_store_epi32(&tmp_result[i][start+16],_res_ij1);
                    _mm512_store_epi32(&tmp_result[i][start+32],_res_ij2);
                    _mm512_store_epi32(&tmp_result[i][start+48],_res_ij3);
                    #endif
                    #endif
                }
            }
           
            #pragma omp barrier
            //stage 2: Process the singly dependent blocks which depend on the independent block
                //i-aligned singly dependent blocks
          
            for(int ib=0;ib<width/BLOCK_WIDTH;++ib){
                //if not the independent block
                if(__builtin_expect(ib!=p,1)){       
                    for(int k=start;k<end;++k){
                        #pragma omp for schedule(static,1) nowait
                        for(int i=start;i<end;++i){
                            #ifndef SIMD
                            int tmp=tmp_result[i][k];
                            for(int j=ib*BLOCK_WIDTH;j<(ib+1)*BLOCK_WIDTH;++j)    tmp_result[i][j]=std::min(tmp_result[i][j],tmp+tmp_result[k][j]);      
                            #else
                            __m512i _tmp=_mm512_set1_epi32(tmp_result[i][k]);  
                            #ifndef UNROLL                       
                            for(int j=ib*BLOCK_WIDTH;j<(ib+1)*BLOCK_WIDTH;j+=16){
                                 __m512i _res_kj=_mm512_load_epi32(&tmp_result[k][j]);
                                 __m512i _res_ij=_mm512_load_epi32(&tmp_result[i][j]);
                                 __m512i _tmp_res_kj=_mm512_add_epi32(_tmp,_res_kj);
                                 _res_ij=_mm512_min_epi32(_res_ij,_tmp_res_kj);                                
                                _mm512_store_epi32(&tmp_result[i][j],_res_ij);
                            }
                            #else
                            __m512i _res_kj0=_mm512_load_epi32(&tmp_result[k][ib*BLOCK_WIDTH]);
                            __m512i _res_kj1=_mm512_load_epi32(&tmp_result[k][ib*BLOCK_WIDTH+16]);
                            __m512i _res_kj2=_mm512_load_epi32(&tmp_result[k][ib*BLOCK_WIDTH+32]);
                            __m512i _res_kj3=_mm512_load_epi32(&tmp_result[k][ib*BLOCK_WIDTH+48]);
                            
                            __m512i _tmp_res_kj0 = _mm512_add_epi32(_tmp,_res_kj0);  
                            __m512i _tmp_res_kj1 = _mm512_add_epi32(_tmp,_res_kj1);
                            __m512i _tmp_res_kj2 = _mm512_add_epi32(_tmp,_res_kj2);
                            __m512i _tmp_res_kj3 = _mm512_add_epi32(_tmp,_res_kj3);

                            __m512i _res_ij0 = _mm512_load_epi32(&tmp_result[i][ib*BLOCK_WIDTH]);
                            __m512i _res_ij1 = _mm512_load_epi32(&tmp_result[i][ib*BLOCK_WIDTH+16]);
                            __m512i _res_ij2 = _mm512_load_epi32(&tmp_result[i][ib*BLOCK_WIDTH+32]);
                            __m512i _res_ij3 = _mm512_load_epi32(&tmp_result[i][ib*BLOCK_WIDTH+48]);

                            _res_ij0 = _mm512_min_epi32(_res_ij0,_tmp_res_kj0);
                            _res_ij1 = _mm512_min_epi32(_res_ij1,_tmp_res_kj1);
                            _res_ij2 = _mm512_min_epi32(_res_ij2,_tmp_res_kj2);
                            _res_ij3 = _mm512_min_epi32(_res_ij3,_tmp_res_kj3);
                            
                            _mm512_store_epi32(&tmp_result[i][ib*BLOCK_WIDTH],_res_ij0);
                            _mm512_store_epi32(&tmp_result[i][ib*BLOCK_WIDTH+16],_res_ij1);
                            _mm512_store_epi32(&tmp_result[i][ib*BLOCK_WIDTH+32],_res_ij2);
                            _mm512_store_epi32(&tmp_result[i][ib*BLOCK_WIDTH+48],_res_ij3);
                            #endif
                            #endif
                        }
                    }
                }

            }
                //j-aligned singly dependent blocks
            for(int jb=0;jb<width/BLOCK_WIDTH;++jb){
                //if not the independent block
                if(__builtin_expect(jb!=p,1)){
                    for(int k=start;k<end;++k){
                        #pragma omp for schedule(static,1) nowait
                        for(int i=jb*BLOCK_WIDTH;i<(jb+1)*BLOCK_WIDTH;++i){
                            #ifndef SIMD
                            int tmp=tmp_result[i][k];
                            for(int j=start;j<end;++j)    tmp_result[i][j]=std::min(tmp_result[i][j],tmp+tmp_result[k][j]);      
                            #else                           
                            __m512i _tmp=_mm512_set1_epi32(tmp_result[i][k]);  
                            #ifndef UNROLL                        
                            for(int j=start;j<end;j+=16){
                                __m512i _res_kj=_mm512_load_epi32(&tmp_result[k][j]);
                                __m512i _res_ij=_mm512_load_epi32(&tmp_result[i][j]);
                                __m512i _tmp_res_kj=_mm512_add_epi32(_tmp,_res_kj);
                                _res_ij=_mm512_min_epi32(_res_ij,_tmp_res_kj);
                                _mm512_store_epi32(&tmp_result[i][j],_res_ij);
                            }
                            #else
                            __m512i _res_kj0=_mm512_load_epi32(&tmp_result[k][start]);
                            __m512i _res_kj1=_mm512_load_epi32(&tmp_result[k][start+16]);
                            __m512i _res_kj2=_mm512_load_epi32(&tmp_result[k][start+32]);
                            __m512i _res_kj3=_mm512_load_epi32(&tmp_result[k][start+48]);
                            
                            __m512i _tmp_res_kj0 = _mm512_add_epi32(_tmp,_res_kj0);  
                            __m512i _tmp_res_kj1 = _mm512_add_epi32(_tmp,_res_kj1);
                            __m512i _tmp_res_kj2 = _mm512_add_epi32(_tmp,_res_kj2);
                            __m512i _tmp_res_kj3 = _mm512_add_epi32(_tmp,_res_kj3);

                            __m512i _res_ij0 = _mm512_load_epi32(&tmp_result[i][start]);
                            __m512i _res_ij1 = _mm512_load_epi32(&tmp_result[i][start+16]);
                            __m512i _res_ij2 = _mm512_load_epi32(&tmp_result[i][start+32]);
                            __m512i _res_ij3 = _mm512_load_epi32(&tmp_result[i][start+48]);

                            _res_ij0 = _mm512_min_epi32(_res_ij0,_tmp_res_kj0);
                            _res_ij1 = _mm512_min_epi32(_res_ij1,_tmp_res_kj1);
                            _res_ij2 = _mm512_min_epi32(_res_ij2,_tmp_res_kj2);
                            _res_ij3 = _mm512_min_epi32(_res_ij3,_tmp_res_kj3);
                            
                            _mm512_store_epi32(&tmp_result[i][start],_res_ij0);
                            _mm512_store_epi32(&tmp_result[i][start+16],_res_ij1);
                            _mm512_store_epi32(&tmp_result[i][start+32],_res_ij2);
                            _mm512_store_epi32(&tmp_result[i][start+48],_res_ij3);
                            #endif
                            #endif

                        }
                    }
                }

            }
           
            #pragma omp barrier

            //stage 3: Process all the other  blocks(which are all doubly dependent)
           
           
            int index[16]={0,64,128,192,256,320,384,448,512,576,640,704,768,832,896,960};
            __m512i _index=_mm512_load_epi32(index);
            int tmp_kj[BLOCK_WIDTH][BLOCK_WIDTH];       
            #pragma omp for schedule(dynamic,1) collapse(2)
            for(int ib=0;ib<width/BLOCK_WIDTH;++ib){ 
                for(int jb=0;jb<width/BLOCK_WIDTH;++jb){  
                    #ifndef SIMD
                    for(int k=start;k<end;++k){
                        for(int j=ib*BLOCK_WIDTH;j<(ib+1)*BLOCK_WIDTH;++j){
                        tmp_kj[j-ib*BLOCK_WIDTH][k-start]=tmp_result[k][j];
                        }
                    }
                    #else 
                    for(int k=start;k<end;++k){
                        for(int j=ib*BLOCK_WIDTH;j<(ib+1)*BLOCK_WIDTH;j+=16){
                            __m512i _kj=_mm512_load_epi32(&tmp_result[k][j]);
                            _mm512_i32scatter_epi32(&tmp_kj[j-ib*BLOCK_WIDTH][k-start],_index,_kj,4);
                        }
                    }
                    #endif
                    // if not the previously processed blocks                   
                    if(__builtin_expect(ib!=p&&jb!=p,1)){                                                                                
                        for(int i=jb*BLOCK_WIDTH;i<(jb+1)*BLOCK_WIDTH;++i){                           
                            for(int j=ib*BLOCK_WIDTH;j<(ib+1)*BLOCK_WIDTH;++j){  
                                #ifndef SIMD
                                int tmp=tmp_result[i][j];  
                                for(int k=start;k<end;++k){                     
                                    tmp=std::min(tmp,tmp_result[i][k]+tmp_result[k][j]);
                                }
                                tmp_result[i][j]=tmp;
                                #else                                 
                                __m512i _tmp=_mm512_set1_epi32(tmp_result[i][j]);  
                                #ifndef UNROLL
                                for(int k=start;k<end;k+=16){  
                                    __m512i _res_kj=_mm512_load_epi32(&tmp_kj[j-ib*BLOCK_WIDTH][k-start]);
                                    __m512i _res_ik=_mm512_load_epi32(&tmp_result[i][k]);
                                    __m512i _res_ik_kj=_mm512_add_epi32(_res_ik,_res_kj);
                                    _tmp=_mm512_min_epi32(_tmp,_res_ik_kj);
                                }
                                tmp_result[i][j]=_mm512_reduce_min_epi32(_tmp);
                                #else
                              
                                __m512i _res_kj0=_mm512_load_epi32(&tmp_kj[j-ib*BLOCK_WIDTH][0]);
                                __m512i _res_kj1=_mm512_load_epi32(&tmp_kj[j-ib*BLOCK_WIDTH][16]);
                                __m512i _res_ik0=_mm512_load_epi32(&tmp_result[i][start]);
                                __m512i _res_ik1=_mm512_load_epi32(&tmp_result[i][start+16]);
                                __m512i _res_ik_kj0=_mm512_add_epi32(_res_ik0,_res_kj0); 
                                __m512i _res_ik_kj1=_mm512_add_epi32(_res_ik1,_res_kj1);
                                _res_ik_kj0=_mm512_min_epi32(_res_ik_kj0,_res_ik_kj1);

                                __m512i _res_kj2=_mm512_load_epi32(&tmp_kj[j-ib*BLOCK_WIDTH][32]);
                                __m512i _res_kj3=_mm512_load_epi32(&tmp_kj[j-ib*BLOCK_WIDTH][48]);                           
                                __m512i _res_ik2=_mm512_load_epi32(&tmp_result[i][start+32]);
                                __m512i _res_ik3=_mm512_load_epi32(&tmp_result[i][start+48]);                                   
                                __m512i _res_ik_kj2=_mm512_add_epi32(_res_ik2,_res_kj2);
                                __m512i _res_ik_kj3=_mm512_add_epi32(_res_ik3,_res_kj3);                                   
                                _res_ik_kj2=_mm512_min_epi32(_res_ik_kj2,_res_ik_kj3);


                                _res_ik_kj0=_mm512_min_epi32(_res_ik_kj0,_res_ik_kj2);


                                //segmentation fault
                                _tmp=_mm512_min_epi32(_res_ik_kj0,_tmp);                 


                                tmp_result[i][j]=_mm512_reduce_min_epi32(_tmp);
                                #endif

                                #endif
                            }
                        }
                    }
                }
            }               
                     
        

        }       
    


        //copy back
        #pragma omp for collapse(2)
        for(int i=0;i<width;++i){
            for(int j=0;j<width;++j){
                result(i,j)=tmp_result[i][j];
            }
        }

    
    }

   
    return result;

}

