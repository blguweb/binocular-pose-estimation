#include <iostream>
#include<fstream>
#include<time.h>
#include<assert.h> 
#include<bits/stdc++.h>

// SIMD部分 
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2

//pthread部分
#include<pthread.h>
#include <semaphore.h>
const int THREAD_NUM = 4;
sem_t	sem_parent,sem_children[THREAD_NUM];  //创建barrier进行阻塞
int cur_iterator = 0; //pthread所用 
using namespace std;

//patchmatch所需要的所有参数集合 
const int N = 1000, M = 4, C = 3, k_iterator = 1, cnt = 1; //一个图像的patch是 MxM 大小的！！！
float img1[C][N][N], img2[C][N][N], ans[C][N][N], foravx1[48*N*N], foravx2[48*N*N];
int rows, cols, prows, pcols, temp;
pair<int, int> f[N-M][N-M];
float error[N-M][N-M];
float w, alpha = 0.5; //随机化搜索的参数 
ofstream ou, proc;
string dir = "para_answers";



void write_proc(int pro){
	proc.open("resources/data/"+dir+"/time.txt");
	proc << pro;
	proc.close();
}

void imread(string path, bool first){
	ifstream in;
	in.open(path);
	in >> rows >> cols;
	prows = rows - M; pcols = cols - M;
	w = max(prows, pcols);
	for(int c=0;c<3;c++){
		for(int i=0;i<rows;i++){
			for(int k=0;k<cols;k++) {
				in >> temp;
				if(first) img1[c][i][k] = temp;
				else img2[c][i][k] = temp;
			}
		}
	}
	in.close();
}

int cal_D(int x1, int y1, int x2, int y2) { //这是计算2范数的！！！ 
	__m256 t1, t2, t3;
	t3 = _mm256_set1_ps(0);
	int id1 = 48*(x1*pcols+y1), id2 = 48*(x2*pcols+y2);
	for(int i=0;i<6;i++){
		t1 = _mm256_loadu_ps(foravx1 + id1 + 8*i);
		t2 = _mm256_loadu_ps(foravx2 + id2 + 8*i);
		t1 = _mm256_sub_ps(t1, t2);
		t1 = _mm256_mul_ps(t1, t1);
		t3 = _mm256_add_ps(t1, t3);
	}
	t3 = _mm256_hadd_ps(t3, t3);
	t3 = _mm256_hadd_ps(t3, t3);
	return t3[0]+t3[4];
}

void initiate_avx(){
	#pragma omp parallel for num_threads(THREAD_NUM)
	for(int i=0;i<prows;i++){
		for(int k=0;k<pcols;k++) { //将img三通道给展开适合avx运算！！！ 
			int id=48*( i*pcols+k );
			for(int p=0;p<M;p++){
				for(int q=0;q<M;q++){
					int id1 = p*M+q;
					foravx1[id+3*id1]=img1[0][i+p][k+q];
					foravx1[id+3*id1+1]=img1[1][i+p][k+q];
					foravx1[id+3*id1+2]=img1[2][i+p][k+q];
					foravx2[id+3*id1]=img2[0][i+p][k+q];
					foravx2[id+3*id1+1]=img2[1][i+p][k+q];
					foravx2[id+3*id1+2]=img2[2][i+p][k+q];
				}
			}
		}	
	}
	#pragma omp barrier
}

void initiate_f() {
	//边界赋值
	for (int i = 0; i < prows; i++) {
		f[i][0] = { i, 0 };
		error[i][0] = cal_D(i, 0, i, 0);
		f[i][pcols - 1] = { i, pcols - 1 };
		error[i][pcols - 1] = cal_D(i, pcols - 1, i, pcols - 1);
	}
	for (int i = 0; i < pcols; i++) {
		f[0][i] = { 0, i };
		error[0][i] = cal_D(0, i, 0, i);
		f[prows - 1][i] = { prows - 1,i };
		error[prows - 1][i] = cal_D(prows - 1, i, prows - 1, i);
	}
	#pragma omp parallel for num_threads(THREAD_NUM)
	for (int i = 1; i < prows - 1; i++) {
		for (int k = 1; k < pcols - 1; k++) {
			int x = i, y = max(1,k-2); //x can not change!!!
			f[i][k] = { x, y };
			error[i][k] = cal_D(i, k, x, y);
		}
	}
	#pragma omp barrier
}

//随机化搜索函数 
void random_search(int i, int k) {
	float radius = 70; //only r = +-70
	int x_match = f[i][k].first, y_match = f[i][k].second;
	for (; radius > 1; radius *= alpha) {
		float weight1 = (float)(rand() % 2000) / 1000 - 1;
		float weight2 = (float)(rand() % 2000) / 1000 - 1;
		int x = x_match + radius * weight1, y = y_match - abs(radius * weight2);
		if (x < 0 || x >= prows || y < 0 || y >= pcols) continue;
		//计算距离
		int d = cal_D(i, k, x, y);
		if (d < error[i][k]) {
			f[i][k] = { x,y };
			error[i][k] = d;
			x_match = x;
			y_match = y;
		}
	}
}

//bool pure(int i, int k){
//	int count = 0;
//	for(int p=0;p<M;p++){
//		for(int q=0;q<M;q++){
//			if(img1[0][i+p][k+q] != 0) count++;
//		}
//	}
//	return count <= 1;
//}
// pthread所用！！！ 
struct threadParm_t{
	int id;
};
//这是线程函数！！！
void *threadFunc(void *parm){
	threadParm_t *p = (threadParm_t *) parm;
	int my_id = p->id, cur_cnt = 0;
	int my_start, my_end, every;
	bool rev = false;
	while(true){
		sem_wait(&sem_children[my_id]);
		if(cur_iterator == 0){ //重置我负责的范围 
			every = (prows-1)/THREAD_NUM;
			my_start = 1 + every * my_id, my_end = my_id==THREAD_NUM-1?prows-1:1+every*(my_id+1);
		}
		for (int i = my_start; i < my_end; i++) {
			for (int k = 1; k < pcols - 1; k++) {
				int start = f[i][k-1].second;
//				if(pure(i, k)) start = k-100;
//				else start = f[i][k-2].second;
//				if(start > k-1) start = k - 20;
				for(int j=max(0, k-30);j<=max(k-10, 0);j++){
					int d = cal_D(i, k, i, j);
					if (d < error[i][k]) {
						error[i][k] = d;
						f[i][k] = {i, j};
					}
				}
//				//计算比较f(ii,kk) f(ii-1,kk) f(ii,kk-1)
//				int d = cal_D(i, k, f[i - 1][k].first, f[i - 1][k].second);
//				if (d < error[i][k]) {
//					error[i][k] = d;
//					f[i][k] = f[i - 1][k];
//				}
//				d = cal_D(i, k, f[i][k - 1].first, f[i][k - 1].second);
//				if (d < error[i][k]) {
//					error[i][k] = d;
//					f[i][k] = f[i][k - 1];
//				}
//				random_search(i, k);
			}
		}
//		if(!rev){ //前向传播 
//			for (int i = my_start; i < my_end; i++) {
//				for (int k = 1; k < pcols - 1; k++) {
//					//计算比较f(ii,kk) f(ii-1,kk) f(ii,kk-1)
//					int d = cal_D(i, k, f[i - 1][k].first, f[i - 1][k].second);
//					if (d < error[i][k]) {
//						error[i][k] = d;
//						f[i][k] = f[i - 1][k];
//					}
//					d = cal_D(i, k, f[i][k - 1].first, f[i][k - 1].second);
//					if (d < error[i][k]) {
//						error[i][k] = d;
//						f[i][k] = f[i][k - 1];
//					}
//					random_search(i, k);
//				}
//			}
//		}
//		else{ //反向传播 
//			for (int i = my_end-1; i >= my_start; i--) {
//				for (int k = pcols - 2; k >= 1; k--) {
//					//计算比较f(ii,kk) f(ii-1,kk) f(ii,kk-1)
//					int d = cal_D(i, k, f[i + 1][k].first, f[i + 1][k].second);
//					if (d < error[i][k]) {
//						error[i][k] = d;
//						f[i][k] = f[i + 1][k];
//					}
//					d = cal_D(i, k, f[i][k + 1].first, f[i][k + 1].second);
//					if (d < error[i][k]) {
//						error[i][k] = d;
//						f[i][k] = f[i][k + 1];
//					}
//					random_search(i, k);
//				}
//			}
//		}
		if(cur_iterator == k_iterator-1) cur_cnt++;
		if(cur_cnt == cnt){
			sem_post(&sem_parent);
			break;
		}
		//rev = ~rev;  only + propagation
		sem_post(&sem_parent);
	}
	pthread_exit(NULL);
}
void forward(){ //pthread合并之后最后只前向，不进行random search 
	for (int i = 1; i < prows - 1; i++) {
		for (int k = 1; k < pcols - 1; k++) {
			//计算比较f(ii,kk) f(ii-1,kk) f(ii,kk-1)
			int d = cal_D(i, k, f[i - 1][k].first, f[i - 1][k].second);
			if (d < error[i][k]) {
				error[i][k] = d;
				f[i][k] = f[i - 1][k];
			}
			d = cal_D(i, k, f[i][k - 1].first, f[i][k - 1].second);
			if (d < error[i][k]) {
				error[i][k] = d;
				f[i][k] = f[i][k - 1];
			}
		}
	}
}

void visualize_result() {
	memset(ans, 0, sizeof(ans));
	float avg = 1.0*M*M;
	#pragma omp parallel for num_threads(THREAD_NUM) 
	for(int i=0;i<prows;i++){
		for(int k=0;k<pcols;k++) { //每个patch对应相加 
			int x=f[i][k].first, y=f[i][k].second;
			for(int p=0;p<M;p++){
				for(int q=0;q<M;q++){
					ans[0][i+p][k+q] += img2[0][x+p][y+q] / avg;
					ans[1][i+p][k+q] += img2[1][x+p][y+q] / avg;
					ans[2][i+p][k+q] += img2[2][x+p][y+q] / avg;
				}
			}
		}	
	}
	#pragma omp barrier
}

//开始执行patchmatch算法 
void para_begin_match() {
	clock_t start,end;
	start = clock();
	initiate_avx(); 
	initiate_f();
	end = clock();
	ou << end-start << " ";
	
	start = clock();
	//创建好所有线程 
	pthread_t thread[THREAD_NUM]; 
	threadParm_t threadParm[THREAD_NUM];  //每个线程自己的结构体 
	sem_init(&sem_parent, 0, 0);
	for(int m=0;m<THREAD_NUM;m++) sem_init(&sem_children[m], 0, 0);
	for(int i=0;i<THREAD_NUM;i++){
		threadParm[i].id = i;
		pthread_create(&thread[i], NULL, threadFunc, (void *)&threadParm[i]);
	}
	for (cur_iterator = 0; cur_iterator < k_iterator; cur_iterator++) {
		for(int m=0;m<THREAD_NUM;m++) sem_post(&sem_children[m]); //唤醒每个线程 
		for(int m=0;m<THREAD_NUM;m++) sem_wait(&sem_parent); //barrier
		write_proc(19*(cur_iterator+1)); 
	}
	forward(); //最后整体前向传播一波 
	end = clock();
	ou << end-start << " ";
	
	start = clock();
	visualize_result(); //reconstruct 图片 
	end = clock();
	ou << end-start << " ";
	write_proc(100);
}

void write_ans(string path){
	ofstream in;
	in.open(path);
	in << rows <<" " << cols << " ";
	for(int i=0;i<rows;i++){
		for(int k=0;k<cols;k++) {
			in << f[i][k].first << " " << f[i][k].second << " ";
		}
	}
	in.close();
}

int main(){
	srand(time(NULL));
	ou.open("resources/data/"+dir+"/para_time_consume.txt");
	for(int cur = 1;cur <= cnt;cur++){ //时间记录如下  f iterations reconstructions total
		//读取两张图片！ 
		clock_t start,end;
		start = clock();
		imread("resources/data/img2txts/A"+to_string(cur)+".txt", 1);
		imread("resources/data/img2txts/B"+to_string(cur)+".txt", 0);
		para_begin_match(); //patchmatch主函数 
		write_ans("resources/data/"+dir+"/ans"+to_string(cur)+".txt"); //写回结果 
		end = clock();
		cout << "Total(ms): " << end-start << endl;
		ou << end-start << endl;
	}
	ou.close();
	return 0;
}
	
