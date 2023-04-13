from flask_restful import Resource
from flask import request
import pandas as pd

class NLPExplainerComparison(Resource):

 
    def post(self):
        params = request.json
        if params is None:
            return "The json body is missing."
        
        #Check params
        if("explainers" not in params):
            return "The explainers to comapre were not specified."

        explainers=params["explainers"]
        if not (isinstance(explainers,list) and len(explainers)==2):
            return "The explainers parameters must be an array containing the names of the two explainers to compare."


        return self.compare(explainers[0], explainers[1])


    def compare(self,exp1,exp2):
        
        df = pd.read_csv('explainers.csv', delimiter=';') 
        matrix = pd.read_csv('detail_with_weight.csv', delimiter=',')
        matrix = matrix.set_index('explainer')

        def format_text(row):
            formatted = ""
            if '#' in row:
                formatted = row.split("#",1)[1]
            else:
                formatted = row.rsplit("/",1)[1]
        
            return formatted

        def simple_format_text(row):
             return format_text(row)[:-1]#.replace("_", "/" )

        def complex_format_text(row, isBackend=False):
            row_list = list(row.replace('[','').replace(']','').replace(' ','').split(','))
            formatted = list()
            for i in row_list:
                formatted.append(format_text(i))
    
            if isBackend == True: 
                formatted = [x[:-1] for x in formatted]

            return formatted

        def array_complex_format_text(row):
            row_list = row[:-2].replace('[','').split('], ')
            formatted = list()
            for elem in row_list:
                formatted.append(complex_format_text(elem))
            return formatted

        def getRow(df, explainer):
            return df.loc[df['Explainer'] == explainer].values.flatten().tolist()

        def getExplanationComplex(explainer1_atts, explainer2_atts, explanation_original, explanationToInclude, isBackend=False):
            common_props = [x for x in explainer1_atts if x in explainer2_atts]
            common_props_len = len(common_props)
            if common_props_len > 0:
                if isBackend==False:
                    explanation_original = explanation_original + explanationToInclude + common_props[0] + ", "
                else:
                    explanation_original = explanation_original + explanationToInclude + ','.join(common_props) + ", "
            return explanation_original

        def getExplanationComplexArray(explainer1_atts, explainer2_atts):
            attributes_list = list()
            for attrib_list1 in explainer1_atts:
                for attrib_list2 in explainer2_atts:
                    common_props = [x for x in attrib_list1 if x in attrib_list2]
                    common_props_len = len(common_props)
                    if common_props_len > 0:
                        attributes_list.append(common_props[0])
                
            return list(set(attributes_list))

        def getSimNL(formatted_df, sim_matrix, explainer1, explainer2):
            explanation = ""
            explainer1_atts = getRow(formatted_df, explainer1)
            explainer2_atts = getRow(formatted_df, explainer2)
            if not explainer1_atts:
                return "No info was found for explainer: '" + str(explainer1)+"'" 
            if not explainer2_atts:
                return "No info was found for explainer: '" + str(explainer2) +"'" 
            if(sim_matrix[explainer1][explainer2] != 0):
                # explanation about why they are similar
                explanation = "They are similar because "
                if explainer1_atts[3] == explainer2_atts[3]:
                    explanation = explanation + "they can be applied to the same dataset type: " + explainer2_atts[3] + " data, "
                if explainer1_atts[4] == explainer2_atts[4]:
                    explanation = explanation + "they have the same concurrentness: " + explainer2_atts[4] + ", "
                if explainer1_atts[5] == explainer2_atts[5]:
                    explanation = explanation + "they have the same scope: " + explainer2_atts[5] + ", "
                if explainer1_atts[6] == explainer2_atts[6]:
                    explanation = explanation + "they have the same portability: " + explainer2_atts[6] + ", "
                if explainer1_atts[7] == explainer2_atts[7]:
                    explanation = explanation + "they have the same target type: " + explainer2_atts[7] + ", "
                if explainer1_atts[10] == explainer2_atts[10]:
                    explanation = explanation + "they have the same computational complexity: " + explainer2_atts[10] + ", "
        
                # for the complex ones, if they share one in the array, show
                # if they share more than one, show the most deep (the one in the beginning of the array)
                explanationToInclude = "they are the same explainability technique type: "
                explanation = getExplanationComplex(explainer1_atts[2], explainer2_atts[2], explanation, explanationToInclude)
                explanationToInclude = "they show the same explanation type: "
                explanation = getExplanationComplex(explainer1_atts[9], explainer2_atts[9], explanation, explanationToInclude)
                explanationToInclude = "they use the same backend: "
                explanation = getExplanationComplex(explainer1_atts[13], explainer2_atts[13], explanation, explanationToInclude, True)
        
        
                attributes_list = getExplanationComplexArray(explainer1_atts[8], explainer2_atts[8])
                if len(attributes_list) > 0:
                    explanation = explanation + "they show the explanation with the same output type: " + ','.join(attributes_list) + ", "
                attributes_list = getExplanationComplexArray(explainer1_atts[11], explainer2_atts[11])
                if len(attributes_list) > 0:
                    explanation = explanation + "they are applicable to the same AI method type: " + ','.join(attributes_list) + ", "
                attributes_list = getExplanationComplexArray(explainer1_atts[12], explainer2_atts[12])
                if len(attributes_list) > 0:
                    explanation = explanation + "they are applicable to the same AI task type: " + ','.join(attributes_list) + ", "
            
    
            #else:
            #    explanation = "They are not similar because they "
                # explanation about why they are not similar
            #    if explainer1_atts[3] != explainer2_atts[3]:
            #        explanation = explanation + "are not applied to the same dataset type. " + explainer1 + " is applicable on " + explainer1_atts[1] + "data , while " + explainer2 + " is applicable on " + explainer2_atts[1] + " data"
            #    if explainer1_atts[6] != explainer2_atts[6]:
            #        explanation = explanation + "do not have the same portability. " + explainer1 + " is " + explainer1_atts[6] + ", while " + explainer2 + " is " + explainer2_atts[6]
            #    if explainer1_atts[7] != explainer2_atts[7]:
            #        explanation = explanation + "do not have the same target type" + explainer1 + " has a" + explainer1_atts[7] + "as target type, while " + explainer2 + " has a" + explainer2_atts[7] + "as target type"
        
            return explanation

        formatted_df = df.copy()

        formatted_df['ExplainabilityTechnique'] = formatted_df['ExplainabilityTechnique'].apply(lambda row: simple_format_text(row))
        formatted_df['DatasetType'] = formatted_df['DatasetType'].apply(lambda row: simple_format_text(row))
        formatted_df['Concurrentness'] = formatted_df['Concurrentness'].apply(lambda row: simple_format_text(row))
        formatted_df['Scope'] = formatted_df['Scope'].apply(lambda row: simple_format_text(row))
        formatted_df['Portability'] = formatted_df['Portability'].apply(lambda row: simple_format_text(row))
        formatted_df['TargetType'] = formatted_df['TargetType'].apply(lambda row: simple_format_text(row))
        formatted_df['Complexity'] = formatted_df['Complexity'].apply(lambda row: simple_format_text(row))

        formatted_df['ExplainabilityTechniqueType'] = formatted_df['ExplainabilityTechniqueType'].apply(lambda row: complex_format_text(row))
        formatted_df['ExplanationType'] = formatted_df['ExplanationType'].apply(lambda row: complex_format_text(row))
        formatted_df['Backend'] = formatted_df['Backend'].apply(lambda row: complex_format_text(row, True))

        formatted_df['OutputType'] = formatted_df['OutputType'].apply(lambda row: array_complex_format_text(row))
        formatted_df['AIMethodType'] = formatted_df['AIMethodType'].apply(lambda row: array_complex_format_text(row))
        formatted_df['AITaskType'] = formatted_df['AITaskType'].apply(lambda row: array_complex_format_text(row))

        return getSimNL(formatted_df, matrix, exp1,exp2)

    

