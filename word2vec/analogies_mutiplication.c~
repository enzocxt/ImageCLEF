//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.


//**************************************************************
//code for multiplication method for the accuracy of word2vec. *
//TBIR KULeuven 2016                                           *
//************************************************************** 

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

const long long max_size = 2000;         // max length of strings
const long long N = 1;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size], st2[max_size], st3[max_size], truth[max_size], topic[max_size];
  topic[0] = 0;
	char *bestw[N];
  char file_name[max_size], st[100][max_size];
  float dist, len, bestd[N], vec[max_size];
 	long long words, size, a, b, st1_vpos, st2_vpos, st3_vpos, c, d, cn;
	long total_questions = 0, tp = 0, missing = 0, total_cat_questions = 0;
  char ch;
  float *M;
  char *vocab;
  if (argc < 2) {
    printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
		vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) {
			fread(&M[a + b * size], sizeof(float), 1, f);
		}
		len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);



  while (1) {
    //null arrays
		for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    a = 0;
		//load first string
		scanf("%s", st1);
	
		// if semicolon then new topic
		if(!strcmp(st1, ":") || strcmp(st1, "EXIT") == 0 || feof(stdin)){
			//printf("\nNew topic");
			if(topic[0] != 0) {
				double recall = (double) tp / total_cat_questions;	
				printf("\n%s \t\t\t Questions: %ld TP: %ld \t Missing: %ld \t  Recall@1: %.4f", topic, total_cat_questions, tp, missing, recall);
				//printf("\n%ld %ld", tp, total_cat_questions);
				total_cat_questions = 0; missing = 0; tp = 0;
			}	
			if(strcmp(st1, "EXIT") == 0 || feof(stdin) ) break;
			
			scanf("%s", topic);
			continue;
		}

		total_questions++;
		total_cat_questions++;

		scanf("%s", st2);
		scanf("%s", st3);
		scanf("%s", truth); 		//load ground truth

		//find position in vocabulary for first, second, third and truth word, then find their pos in vocabulary
		for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st1)) break;
		st1_vpos = b;
		if(	st1_vpos == words ) { 
			missing++;
			printf("\nWord %s is not in dictionary", st1);
			continue;
		}
		for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st2)) break;
		st2_vpos = b;
		if(	st2_vpos == words ) { 
			missing++;
			printf("\nWord %s is not in dictionary", st2);
			continue;
		}
		for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st3)) break;
		st3_vpos = b;
		if(	st3_vpos == words ) { 
			missing++;
			printf("\nWord %s is not in dictionary", st3);
			continue;
		}

		// null cosine similarity vector
		for (a = 0; a < size; a++) vec[a] = 0;
   	for (a = 0; a < size; a++) vec[a] += M[a + st3_vpos * size] - M[a + st1_vpos * size] + M[a + st2_vpos * size]; 

		//normalize vector
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    
		for(c = 0; c < words; c++) {
			//skip input words
			if (c == st1_vpos ) continue;
			if (c == st2_vpos ) continue;
			if (c == st3_vpos ) continue;
			if (c > 60000) break;
		      dist = 0;
		      float cad = 0;
		      float cbd = 0;
		      float ccd = 0;
		      
		      for (a = 0; a < size; a++) cad += M[a + st1_vpos * size] * M[a + c * size]; //cos(d',a)     
		      for (a = 0; a < size; a++) cbd += M[a + st2_vpos * size] * M[a + c * size]; //cos(d',b)     
		      for (a = 0; a < size; a++) ccd += M[a + st3_vpos * size] * M[a + c * size]; //cos(d',c)     

		      cad = (cad + 1)/2;
		      cbd = (cbd + 1)/2;
		      ccd = (ccd + 1)/2;

		      if(cad == 0)  cad = 0.001;
		      dist = (ccd * cbd) / cad;
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &vocab[c * max_w]);
          break;
        }
      }

		}
		
			
		if(strcmp(bestw[0], truth) == 0){
			//true positive
			tp++;
		}
		printf("\nResult: %s %s", bestw[0], truth);	

  }

	printf("\nTotal questions: %ld\n", total_questions); 
  return 0;
}
