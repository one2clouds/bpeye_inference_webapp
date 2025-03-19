import torch.nn as nn 
import torch 


def predict_amd_disease(img_path, img_transform, my_transforms, model, device): 

    preprocessed_img = img_transform(img_path)

    # print(torchvision.transforms.functional.pil_to_tensor(preprocessed_img).shape)

    preprocessed_transformed_image = my_transforms(preprocessed_img)

    preprocessed_transformed_image = preprocessed_transformed_image.unsqueeze(0).to(device)
    # print(preprocessed_transformed_image.shape)

    outputs = nn.Sigmoid()(model(preprocessed_transformed_image))

    confidence_score, predicted_test = torch.max(outputs, 1)

    threshold_value = 0.7

    if predicted_test == 0 and confidence_score > threshold_value:
        predicted_label = "Non-AMD"
    else:
        predicted_label = "AMD"

    return predicted_label, confidence_score.item()*100
